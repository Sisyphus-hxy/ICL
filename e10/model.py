"""E10 Learned Low-Rank Transfer Model over Frozen E9 Encoder, Reader & Action Decoder.

Architecture invariants:
  - Frozen E9 Stage 1 Local Teacher (`local_encoder` + `action_decoder`, `requires_grad=False`, `eval()`).
  - Frozen E9 Stage 2 Direct Reader (`SimpleBehaviorReader`, loaded from `e9/outputs/stage2_direct/train/best.pt`,
    `requires_grad=False`, permanently `eval()`).
  - Fixed train-only P0 SVD orthonormal basis `U_K` in R^(3072 x K) (non-trainable buffer).
  - Trainable Transfer backbone (`LowRankTransferTransformer`): identical 3-layer Pre-LN Transformer
    Decoder backbone to `ResidualTransformer` (`e5/model.py`), differing ONLY in the output head
    `coeff_head: Linear(3072 -> K)` and low-rank projection `U_K @ c`.
  - Two modes:
      * `residual` (E10-A/B/C): `z_hat = z_source + U_K @ c` (zero-init -> exact identity at step 0)
      * `control`  (Control)  : `z_hat = z_mean   + U_K @ c` (removes demo residual skip, K=32)
  - NO extra source state input (`h_c`), NO extra auxiliary losses (`L = L_diff + 0.5 * L_z`).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from e5.actions import action_10d_to_7d, extract_exec_actions
from e5.bpp_compat import import_bpp_components
from e5.model import E5Policy
from e7.model import SimpleBehaviorReader, compute_attention_stats
from e9.stage1_local_model import E9Stage1LocalPolicy, load_frozen_e9_stage1_teacher

from . import (
    E9_STAGE1_CKPT,
    E9_STAGE2_CKPT,
    EMBED_DIM,
    EXEC_HORIZON,
    LATENT_DIM,
    LATENT_TOKENS,
    NUM_HEADS,
)

_BPP = import_bpp_components()
bpp_init_weights = _BPP["init_weights"]


class LowRankTransferTransformer(nn.Module):
    """3-layer Transformer Decoder backbone matching `ResidualTransformer` (`e5/model.py`)
    with low-rank coefficient head `c = F(z_source, h_query) in R^K`."""

    def __init__(
        self,
        *,
        rank_k: int,
        embed_dim: int = EMBED_DIM,
        num_layers: int = 3,
        num_heads: int = NUM_HEADS,
        ffn_dim: int = 3072,
        num_query_tokens: int = LATENT_TOKENS,
        dropout: float = 0.1,
        latent_scale_s_z: float = 1.006133,
        init_std: float = 0.0,
    ):
        super().__init__()
        if rank_k < 1 or rank_k > num_query_tokens * embed_dim:
            raise ValueError(f"Invalid rank_k={rank_k}")
        self.rank_k = int(rank_k)
        self.embed_dim = int(embed_dim)
        self.num_query_tokens = int(num_query_tokens)
        self.flat_dim = self.num_query_tokens * self.embed_dim  # 3072
        self.latent_scale_s_z = float(latent_scale_s_z)
        self.init_std = float(init_std)
        # Scale so RMS(U_K @ c) = s_z * RMS(c_raw) across all K (matches ResidualTransformer)
        self.coeff_scale = float(self.latent_scale_s_z * math.sqrt(self.flat_dim / float(self.rank_k)))

        layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=int(num_heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.ref_in_norm = nn.LayerNorm(self.embed_dim)
        self.query_in_norm = nn.LayerNorm(self.embed_dim)
        self.decoder = nn.TransformerDecoder(decoder_layer=layer, num_layers=int(num_layers))
        self.ref_pos_emb = nn.Parameter(torch.randn(1, self.num_query_tokens, self.embed_dim))
        self.query_pos_emb = nn.Parameter(torch.randn(1, self.num_query_tokens, self.embed_dim))
        self.out_norm = nn.LayerNorm(self.embed_dim)
        self.coeff_head = nn.Linear(self.flat_dim, self.rank_k)

        self.init_weights()

    def init_weights(self) -> None:
        self.decoder.apply(bpp_init_weights)
        for norm in (self.ref_in_norm, self.query_in_norm, self.out_norm):
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)
        nn.init.normal_(self.ref_pos_emb, mean=0.0, std=0.02)
        nn.init.normal_(self.query_pos_emb, mean=0.0, std=0.02)
        if self.init_std > 0.0:
            nn.init.normal_(self.coeff_head.weight, mean=0.0, std=self.init_std)
        else:
            nn.init.zeros_(self.coeff_head.weight)
        nn.init.zeros_(self.coeff_head.bias)

    def forward(self, z_source: torch.Tensor, h_q: torch.Tensor) -> torch.Tensor:
        """Map `(z_source, h_q)` `(B, 4, 768)` -> scaled low-rank coefficients `c` `(B, K)`."""
        if z_source.shape != h_q.shape:
            raise ValueError(
                f"z_source {tuple(z_source.shape)} and h_q {tuple(h_q.shape)} must match"
            )
        h = self.decoder(
            tgt=self.ref_in_norm(z_source) + self.ref_pos_emb,
            memory=self.query_in_norm(h_q) + self.query_pos_emb,
        )
        h_flat = self.out_norm(h).reshape(h.shape[0], self.flat_dim)
        c_raw = self.coeff_head(h_flat)
        return c_raw * self.coeff_scale


class E10LowRankPolicy(nn.Module):
    """P1 Learned Low-Rank Transfer Policy with frozen E9 Teacher, frozen E9 Reader,
    fixed P0 SVD basis `U_K`, and trainable `LowRankTransferTransformer`."""

    def __init__(
        self,
        teacher: E9Stage1LocalPolicy,
        reader: SimpleBehaviorReader,
        *,
        basis_u_k: torch.Tensor,
        z_mean: torch.Tensor,
        latent_scale_s_z: float,
        transfer_mode: str = "residual",
        residual_layers: int = 3,
        residual_heads: int = NUM_HEADS,
        residual_ffn_dim: int = 3072,
        residual_dropout: float = 0.1,
        control_init_std: float = 0.002,
    ):
        super().__init__()
        if transfer_mode not in ("residual", "control"):
            raise ValueError(f"Unsupported transfer_mode={transfer_mode!r}")
        if basis_u_k.ndim != 2 or basis_u_k.shape[0] != LATENT_DIM:
            raise ValueError(f"Expected basis_u_k ({LATENT_DIM}, K), got {tuple(basis_u_k.shape)}")

        u_f = basis_u_k.detach().float().contiguous()
        gram = u_f.T @ u_f
        ortho_err = float((gram - torch.eye(u_f.shape[1], dtype=torch.float32)).abs().max().item())
        if ortho_err > 1e-3:
            raise ValueError(f"basis_u_k is not orthonormal: max |U^T U - I| = {ortho_err:.3e}")

        self.transfer_mode = str(transfer_mode)
        self.prediction_mode = self.transfer_mode  # Compatibility alias for D1-D4 / servers
        self.rank_k = int(u_f.shape[1])

        self.teacher = teacher
        self.reader = reader
        self._freeze_teacher_and_reader()

        self.register_buffer("basis_u_k", u_f)  # (3072, K) float32
        self.register_buffer(
            "latent_scale_s_z",
            torch.tensor(float(latent_scale_s_z), dtype=torch.float32),
        )

        zm = z_mean.detach().float().reshape(1, LATENT_TOKENS, EMBED_DIM).contiguous()
        self.register_buffer("z_mean", zm)

        init_std = float(control_init_std) if self.transfer_mode == "control" else 0.0
        self.transfer = LowRankTransferTransformer(
            rank_k=self.rank_k,
            embed_dim=EMBED_DIM,
            num_layers=residual_layers,
            num_heads=residual_heads,
            ffn_dim=residual_ffn_dim,
            num_query_tokens=LATENT_TOKENS,
            dropout=residual_dropout,
            latent_scale_s_z=float(latent_scale_s_z),
            init_std=init_std,
        )

    def _freeze_teacher_and_reader(self) -> None:
        for mod in (self.teacher, self.reader):
            for p in mod.parameters():
                p.requires_grad_(False)
            mod.eval()

    def train(self, mode: bool = True) -> "E10LowRankPolicy":
        """Keep frozen Teacher AND frozen Reader permanently in eval() mode."""
        super().train(mode)
        self.teacher.eval()
        self.reader.eval()
        return self

    def trainable_parameters(self) -> List[nn.Parameter]:
        """Strictly ONLY `self.transfer` parameters are trainable."""
        params = [p for p in self.transfer.parameters() if p.requires_grad]
        # Defensive invariant check
        frozen_trainable = sum(
            p.numel()
            for mod in (self.teacher, self.reader)
            for p in mod.parameters()
            if p.requires_grad
        )
        if frozen_trainable != 0:
            raise RuntimeError(f"Frozen teacher/reader has {frozen_trainable} trainable params!")
        return params

    def get_optim_groups(self, lr: float, weight_decay: float) -> List[Dict[str, Any]]:
        """Exact AdamW decay / no-decay grouping matching E7/E9 Stage 2."""
        decay, no_decay = [], []
        no_decay_kw = ("bias", "norm", "pos_emb")
        for name, p in self.transfer.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or any(kw in name for kw in no_decay_kw):
                no_decay.append(p)
            else:
                decay.append(p)
        return [
            {"params": decay, "lr": float(lr), "weight_decay": float(weight_decay)},
            {"params": no_decay, "lr": float(lr), "weight_decay": 0.0},
        ]

    @torch.no_grad()
    def read_cross_latent(
        self,
        h_mem: torch.Tensor,
        b_mem: torch.Tensor,
        mem_pad_mask: Optional[torch.Tensor],
        query_rgb_feats: torch.Tensor,
        *,
        h_q: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Frozen E9 Reader retrieval `z_source = sum_i w_i * b_i` (strictly no grad)."""
        if h_q is None:
            h_q = E5Policy.format_query_tokens(query_rgb_feats)
        z_c, weights = self.reader(h_q, h_mem, b_mem, mem_pad_mask=mem_pad_mask)
        return z_c, h_q, weights

    def adapt_latent(
        self,
        z_source: torch.Tensor,
        h_q: torch.Tensor,
        *,
        delta_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute low-rank adapted latent `z_hat`:
          - `residual`: `z_hat = z_source + delta_scale * (U_K @ c)`
          - `control` : `z_hat = z_mean   + (U_K @ c)` (at delta_scale == 0 returns z_source for D2)
        Returns `(z_hat, z_hat - z_source)` for full compatibility with E9 diagnostics.
        """
        scale = float(delta_scale)
        if scale == 0.0:
            return z_source, torch.zeros_like(z_source)
        if self.transfer_mode == "control" and scale != 1.0:
            raise ValueError(f"delta_scale must be 1.0 or 0.0 in control mode, got {scale}")

        B = z_source.shape[0]
        c = self.transfer(z_source, h_q)  # (B, K)
        # Project c through fixed orthonormal basis U_K in float32 for precision, then match dtype
        u_k = self.basis_u_k  # (3072, K) float32
        lowrank_vec = torch.matmul(c.float(), u_k.T).reshape(B, LATENT_TOKENS, EMBED_DIM)
        lowrank_vec = lowrank_vec.to(dtype=z_source.dtype)

        if self.transfer_mode == "residual":
            if scale != 1.0:
                lowrank_vec = lowrank_vec * scale
            z_hat = z_source + lowrank_vec
            return z_hat, lowrank_vec

        # Control mode: z_hat = z_mean + U_K @ c (no z_source skip)
        zm = self.z_mean.to(device=z_source.device, dtype=z_source.dtype)
        z_hat = zm + lowrank_vec
        return z_hat, z_hat - z_source

    def forward(
        self,
        h_mem: torch.Tensor,
        b_mem: torch.Tensor,
        mem_pad_mask: Optional[torch.Tensor],
        query_rgb_feats: torch.Tensor,
        z_self: torch.Tensor,
        actions_10d: torch.Tensor,
        is_self: Optional[torch.Tensor] = None,
        *,
        lambda_a: float = 1.0,
        lambda_z: float = 0.5,
        noise: Optional[torch.Tensor] = None,
        pertub_noise: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Exact E9 Stage 2 loss: L = lambda_a * L_diff(A_q | z_hat) + lambda_z * L_z.
        NO extra auxiliary losses; frozen Reader runs strictly under torch.no_grad()."""
        h_q = E5Policy.format_query_tokens(query_rgb_feats).detach()
        h_mem_det = h_mem.detach()
        b_mem_det = b_mem.detach()
        z_s = z_self.detach()
        s_z = self.latent_scale_s_z.to(device=h_q.device, dtype=torch.float32).clamp_min(1e-6)

        with torch.no_grad():
            z_c, _, attn_w = self.read_cross_latent(
                h_mem_det, b_mem_det, mem_pad_mask, query_rgb_feats, h_q=h_q
            )
        z_c = z_c.detach()

        z_hat, delta_z = self.adapt_latent(z_c, h_q)

        per_sample_lz = ((z_hat.float() - z_s.float()) / s_z).square().mean(dim=(1, 2))
        latent_loss_norm = per_sample_lz.mean()

        if lambda_a != 0.0:
            per_sample_diff = self.teacher.diffusion_loss_from_z_self(
                z_hat,
                actions_10d,
                noise=noise,
                pertub_noise=pertub_noise,
                timesteps=timesteps,
            )
            diff_loss = per_sample_diff.mean()
        else:
            diff_loss = torch.zeros((), device=z_hat.device, dtype=torch.float32)

        total_loss = lambda_a * diff_loss + lambda_z * latent_loss_norm

        with torch.no_grad():
            B = z_hat.shape[0]
            zc_f = z_c.float()
            zq_f = z_hat.detach().float()
            zs_f = z_s.float()
            dz_f = delta_z.detach().float()

            latent_mse_raw = (zq_f - zs_f).square().mean()
            nodelta_mse_raw = (zc_f - zs_f).square().mean()
            nodelta_lz = ((zc_f - zs_f) / s_z).square().mean()
            cos_nodelta = F.cosine_similarity(zc_f.reshape(B, -1), zs_f.reshape(B, -1), dim=-1).mean()
            cos_adapted = F.cosine_similarity(zq_f.reshape(B, -1), zs_f.reshape(B, -1), dim=-1).mean()
            gap = zs_f - zc_f
            dir_cos = F.cosine_similarity(dz_f.reshape(B, -1), gap.reshape(B, -1), dim=-1).mean()
            delta_sq = dz_f.square().mean(dim=(1, 2))
            delta_rms = torch.sqrt(delta_sq.mean() + 1e-12)
            z_cross_rms = torch.sqrt(zc_f.square().mean() + 1e-12)

            if is_self is not None:
                is_self_b = is_self.to(device=z_hat.device, dtype=torch.bool)
                s_m = is_self_b.float()
                c_m = (~is_self_b).float()
                n_s = s_m.sum()
                n_c = c_m.sum()
                self_delta_rms = torch.sqrt((delta_sq * s_m).sum() / n_s.clamp_min(1.0) + 1e-12)
                cross_delta_rms = torch.sqrt((delta_sq * c_m).sum() / n_c.clamp_min(1.0) + 1e-12)
            else:
                n_s = torch.tensor(0.0, device=z_hat.device)
                n_c = torch.tensor(float(B), device=z_hat.device)
                self_delta_rms = torch.tensor(0.0, device=z_hat.device)
                cross_delta_rms = delta_rms

            astats = compute_attention_stats(attn_w, mem_pad_mask)

        return {
            "loss": total_loss,
            "diff_loss": diff_loss.detach(),
            "latent_loss_norm": latent_loss_norm.detach(),
            "latent_mse_raw": latent_mse_raw,
            "nodelta_mse_raw": nodelta_mse_raw,
            "nodelta_loss_norm": nodelta_lz,
            "cos_nodelta_self": cos_nodelta,
            "cos_adapted_self": cos_adapted,
            "dir_cos": dir_cos,
            "delta_rms": delta_rms,
            "cross_delta_rms": cross_delta_rms,
            "self_delta_rms": self_delta_rms,
            "z_cross_rms": z_cross_rms,
            "attn_entropy": astats["entropy"].mean(),
            "attn_norm_entropy": astats["norm_entropy"].mean(),
            "attn_max_weight": astats["max_weight"].mean(),
            "attn_eff_pos": astats["eff_pos_exp"].mean(),
            "attn_eff_ipr": astats["eff_pos_ipr"].mean(),
            "n_self": n_s,
            "n_cross": n_c,
        }

    @torch.inference_mode()
    def predict_action_from_behavior_memory(
        self,
        h_mem: torch.Tensor,
        b_mem: torch.Tensor,
        mem_pad_mask: Optional[torch.Tensor],
        query_rgb_feats: torch.Tensor,
        *,
        apply_delta: bool = True,
        delta_scale: float = 1.0,
        generator: Optional[torch.Generator] = None,
        init_noise: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        z_cross, h_q, attn_weights = self.read_cross_latent(
            h_mem, b_mem, mem_pad_mask, query_rgb_feats
        )
        if apply_delta and delta_scale != 0.0:
            z_cond, delta_z = self.adapt_latent(z_cross, h_q, delta_scale=delta_scale)
        else:
            delta_z = torch.zeros_like(z_cross)
            z_cond = z_cross

        action_10d = self.teacher.sample_from_z_self(
            z_cond, generator=generator, init_noise=init_noise
        )
        return {
            "action_10d": action_10d,
            "action_7d": action_10d_to_7d(action_10d),
            "exec_action_7d": extract_exec_actions(action_10d, exec_horizon=EXEC_HORIZON),
            "z_cross": z_cross,
            "z_cond": z_cond,
            "delta_z": delta_z,
            "attn_weights": attn_weights,
        }

    def parameter_summary(self, visual_encoder_params: int = 171_598_848) -> Dict[str, Any]:
        t_tot = sum(p.numel() for p in self.teacher.parameters())
        t_tr = sum(p.numel() for p in self.teacher.parameters() if p.requires_grad)
        r_tot = sum(p.numel() for p in self.reader.parameters())
        r_tr = sum(p.numel() for p in self.reader.parameters() if p.requires_grad)
        f_tot = sum(p.numel() for p in self.transfer.parameters())
        f_tr = sum(p.numel() for p in self.transfer.parameters() if p.requires_grad)
        return {
            "visual_encoder_frozen": visual_encoder_params,
            "stage1_teacher_frozen": t_tot,
            "stage1_teacher_trainable": t_tr,
            "reader_frozen": r_tot,
            "reader_trainable": r_tr,
            "transfer_total": f_tot,
            "transfer_trainable": f_tr,
            "total_trainable": t_tr + r_tr + f_tr,
            "rank_k": self.rank_k,
            "transfer_mode": self.transfer_mode,
            "latent_scale_s_z": float(self.latent_scale_s_z.item()),
            "z_mean_rms": float(torch.sqrt(self.z_mean.square().mean()).item()),
            "coeff_scale": float(self.transfer.coeff_scale),
        }


def build_e10_policy(
    *,
    rank_k: int,
    transfer_mode: str = "residual",
    basis_path: Path | str,
    basis_key: str = "U_reader",
    stage1_ckpt: Path | str = E9_STAGE1_CKPT,
    e9_stage2_ckpt: Path | str = E9_STAGE2_CKPT,
    device: torch.device | str = "cpu",
) -> Tuple[E10LowRankPolicy, Dict[str, Any]]:
    """Construct E10LowRankPolicy with frozen E9 Teacher, frozen E9 Stage 2 Reader,
    and fixed train-fitted P0 SVD basis `U_K`."""
    teacher = load_frozen_e9_stage1_teacher(stage1_ckpt, device="cpu")
    e9_payload = torch.load(Path(e9_stage2_ckpt), map_location="cpu", weights_only=False)
    e9_cfg = e9_payload.get("config", {})
    s_z = float(e9_payload.get("latent_scale_s_z", e9_cfg["latent_scale_s_z"]))
    z_mean = e9_payload["direct_latent_bias"].float()

    reader = SimpleBehaviorReader(
        num_query_tokens=LATENT_TOKENS,
        embed_dim=EMBED_DIM,
        attn_dim=int(e9_cfg.get("reader_attn_dim", EMBED_DIM)),
    )
    reader.load_state_dict(e9_payload["reader"], strict=True)

    bases = torch.load(Path(basis_path), map_location="cpu", weights_only=False)
    u_full = bases[basis_key].float()  # (3072, >=K)
    if u_full.shape[1] < int(rank_k):
        raise ValueError(f"Basis {basis_key} has {u_full.shape[1]} cols < rank_k={rank_k}")
    basis_u_k = u_full[:, : int(rank_k)].contiguous()

    policy = E10LowRankPolicy(
        teacher=teacher,
        reader=reader,
        basis_u_k=basis_u_k,
        z_mean=z_mean,
        latent_scale_s_z=s_z,
        transfer_mode=transfer_mode,
        residual_layers=int(e9_cfg.get("residual_layers", 3)),
        residual_heads=int(e9_cfg.get("residual_heads", NUM_HEADS)),
        residual_ffn_dim=int(e9_cfg.get("residual_ffn_dim", 3072)),
        residual_dropout=float(e9_cfg.get("residual_dropout", 0.1)),
    )
    meta = {
        "e9_stage1_ckpt": str(Path(stage1_ckpt).resolve()),
        "e9_stage2_ckpt": str(Path(e9_stage2_ckpt).resolve()),
        "e9_stage2_epoch": e9_payload.get("epoch"),
        "basis_path": str(Path(basis_path).resolve()),
        "basis_key": basis_key,
        "basis_version": bases.get("version"),
        "basis_n_train": bases.get("n_train"),
    }
    return policy.to(device), meta


def load_e10_policy(
    ckpt_path: Path | str,
    *,
    stage1_ckpt: Optional[Path | str] = None,
    e9_stage2_ckpt: Optional[Path | str] = None,
    device: torch.device | str = "cpu",
) -> Tuple[E10LowRankPolicy, Dict[str, Any]]:
    """Load saved E10 checkpoint (restoring frozen Teacher, frozen Reader, fixed basis `U_K`,
    and trained `transfer` weights) with strict verification."""
    payload = torch.load(Path(ckpt_path), map_location="cpu", weights_only=False)
    cfg = payload["config"]
    s1_path = Path(stage1_ckpt) if stage1_ckpt is not None else Path(cfg["e9_stage1_ckpt"])
    s2_path = Path(e9_stage2_ckpt) if e9_stage2_ckpt is not None else Path(cfg["e9_stage2_ckpt"])

    teacher = load_frozen_e9_stage1_teacher(s1_path, device="cpu")
    e9_payload = torch.load(s2_path, map_location="cpu", weights_only=False)
    e9_cfg = e9_payload.get("config", {})

    reader = SimpleBehaviorReader(
        num_query_tokens=LATENT_TOKENS,
        embed_dim=EMBED_DIM,
        attn_dim=int(e9_cfg.get("reader_attn_dim", EMBED_DIM)),
    )
    reader.load_state_dict(e9_payload["reader"], strict=True)

    policy = E10LowRankPolicy(
        teacher=teacher,
        reader=reader,
        basis_u_k=payload["basis_u_k"],
        z_mean=payload["z_mean"],
        latent_scale_s_z=float(payload["latent_scale_s_z"]),
        transfer_mode=str(payload["transfer_mode"]),
        residual_layers=int(cfg.get("residual_layers", 3)),
        residual_heads=int(cfg.get("residual_heads", NUM_HEADS)),
        residual_ffn_dim=int(cfg.get("residual_ffn_dim", 3072)),
        residual_dropout=float(cfg.get("residual_dropout", 0.1)),
    )
    policy.transfer.load_state_dict(payload["transfer"], strict=True)
    policy.eval()
    return policy.to(device), payload
