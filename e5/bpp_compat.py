"""Read-only compatibility bridge for reusing official BPP modules and checkpoints."""

from __future__ import annotations

import builtins
import fcntl
import hashlib
import json
import pickle
import sys
import types
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict

import torch

from . import BPP_CKPT, BPP_SRC, OUTPUT_ROOT

# RGB token positions inside BPP's 10-token query sequence:
# Per timestep, TransformerObsEncoder orders keys as:
#   [0: agentview_rgb, 1: eye_in_hand_rgb, 2: ee_ori, 3: ee_pos, 4: gripper_states]
# Across img_obs_horizon=2:
#   t-1 -> indices 0..4 (RGB at 0, 1)
#   t   -> indices 5..9 (RGB at 5, 6)
BPP_QUERY_RGB_POS_INDICES = (0, 1, 5, 6)


def ensure_bpp_submodules_importable() -> None:
    """Register lightweight package stubs so BPP pure-PyTorch modules import cleanly
    without pulling optional dependencies (numcodecs, dill, omegaconf, diffusers)."""
    bpp_src_str = str(BPP_SRC)
    if bpp_src_str not in sys.path:
        sys.path.insert(0, bpp_src_str)

    stub_packages = [
        "behavior_prompting",
        "behavior_prompting.common",
        "behavior_prompting.train_network",
        "behavior_prompting.train_network.model",
        "behavior_prompting.train_network.model.common",
        "behavior_prompting.train_network.model.diffusion",
        "behavior_prompting.train_network.utils",
    ]
    for pkg in stub_packages:
        if pkg not in sys.modules:
            mod = types.ModuleType(pkg)
            mod.__path__ = [str(BPP_SRC / pkg.replace(".", "/"))]
            sys.modules[pkg] = mod


def import_bpp_components():
    """Import official BPP modules directly from bpp/src/behavior_prompting/."""
    ensure_bpp_submodules_importable()
    from behavior_prompting.train_network.model.common.rotation_transformer import (
        RotationTransformer,
    )
    from behavior_prompting.train_network.model.common.transformer_decoder_with_attention import (
        TransformerDecoderLayerWithAttn,
        TransformerDecoderWithAttn,
    )
    from behavior_prompting.train_network.model.diffusion.conditional_unet1d import (
        ConditionalUnet1D,
    )
    from behavior_prompting.train_network.utils.model_util import (
        get_optim_groups,
        init_weights,
    )

    return {
        "ConditionalUnet1D": ConditionalUnet1D,
        "TransformerDecoderLayerWithAttn": TransformerDecoderLayerWithAttn,
        "TransformerDecoderWithAttn": TransformerDecoderWithAttn,
        "RotationTransformer": RotationTransformer,
        "get_optim_groups": get_optim_groups,
        "init_weights": init_weights,
    }


class _DummyUnpickledObj:
    def __init__(self, *args, **kwargs):
        self._args = args
        self._kwargs = kwargs

    def __setstate__(self, state):
        self._state = state


class _BPPCheckpointUnpickler(pickle.Unpickler):
    """Unpickles BPP checkpoints saved with dill + OmegaConf without requiring them."""

    def find_class(self, module: str, name: str):
        if module.startswith("dill"):
            if name == "_load_type":
                return lambda type_name: getattr(builtins, type_name, _DummyUnpickledObj)
            return lambda *args, **kwargs: _DummyUnpickledObj(*args, **kwargs)
        if module.startswith(("omegaconf", "behavior_prompting")):
            return type(f"Stub_{module}_{name}", (_DummyUnpickledObj,), {})
        return super().find_class(module, name)


def _load_raw_bpp_state_dict(ckpt_path: Path) -> Dict[str, torch.Tensor]:
    fake_pickle = types.ModuleType("bpp_safe_pickle")
    fake_pickle.Unpickler = _BPPCheckpointUnpickler
    fake_pickle.load = lambda f, **kw: _BPPCheckpointUnpickler(f, **kw).load()
    payload = torch.load(
        ckpt_path, map_location="cpu", pickle_module=fake_pickle, weights_only=False
    )
    raw_sd = payload["state_dicts"]["model"]
    return {k: v for k, v in raw_sd.items() if isinstance(v, torch.Tensor)}


def _tensor_dict_sha256(tensors: Dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(tensors):
        t = tensors[k].detach().cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def extract_bpp_pretrained_weights(
    ckpt_path: Path = BPP_CKPT,
    cache_path: Path | None = None,
) -> Path:
    """Extract required pretrained tensors from the 6.5 GB BPP checkpoint once into a
    compact file (~900 MB) for fast, deterministic loading across scripts and DDP ranks."""
    ckpt_path = Path(ckpt_path).resolve()
    if cache_path is None:
        cache_path = OUTPUT_ROOT / "bpp_pretrained_weights.pt"
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_path.with_suffix(".lock")

    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        st = ckpt_path.stat()
        source_meta = {
            "source_ckpt": str(ckpt_path),
            "size": st.st_size,
            "mtime_ns": st.st_mtime_ns,
        }
        if cache_path.is_file():
            meta_path = cache_path.with_suffix(".json")
            if meta_path.is_file():
                meta = json.loads(meta_path.read_text())
                if meta.get("source") == source_meta:
                    return cache_path

        sd = _load_raw_bpp_state_dict(ckpt_path)

        # Verify shared prompt/receding vision encoder invariant in BPP checkpoint.
        av_pref = "obs_encoder.obs_encoder.prompt_obs_encoder.key_model_map.agentview_rgb."
        eh_pref = "obs_encoder.obs_encoder.prompt_obs_encoder.key_model_map.eye_in_hand_rgb."
        dec_pref = "obs_encoder.prompt_with_obs_decoder."

        clip_agentview = {
            k[len(av_pref) :]: v.clone() for k, v in sd.items() if k.startswith(av_pref)
        }
        clip_eye_in_hand = {
            k[len(eh_pref) :]: v.clone() for k, v in sd.items() if k.startswith(eh_pref)
        }
        reader_decoder = {
            k[len(dec_pref) :]: v.clone() for k, v in sd.items() if k.startswith(dec_pref)
        }
        if not clip_agentview or not clip_eye_in_hand or not reader_decoder:
            raise RuntimeError(f"Unexpected BPP checkpoint layout at {ckpt_path}")

        prompt_pos_emb = sd["obs_encoder.prompt_pos_emb"].clone()  # (1, 50, 768)
        attention_sink_tokens = sd["obs_encoder.attention_sink_tokens"].clone()  # (1, 4, 768)
        full_query_pos = sd["obs_encoder.prompt_current_obs_pos_emb"].clone()  # (1, 10, 768)
        rgb_query_pos = full_query_pos[:, list(BPP_QUERY_RGB_POS_INDICES), :].clone()  # (1, 4, 768)

        action_norm = {
            "scale": sd["normalizer.params_dict.action.scale"].clone(),
            "offset": sd["normalizer.params_dict.action.offset"].clone(),
            "min": sd["normalizer.params_dict.action.input_stats.min"].clone(),
            "max": sd["normalizer.params_dict.action.input_stats.max"].clone(),
            "mean": sd["normalizer.params_dict.action.input_stats.mean"].clone(),
            "std": sd["normalizer.params_dict.action.input_stats.std"].clone(),
        }

        # Record exact UNet cond_encoder shape from BPP checkpoint for compatibility reporting.
        unet_cond_shape = tuple(sd["model.down_modules.0.0.cond_encoder.1.weight"].shape)
        unet_total_params = sum(v.numel() for k, v in sd.items() if k.startswith("model."))

        bundle: Dict[str, Any] = {
            "source": source_meta,
            "clip_agentview": clip_agentview,
            "clip_eye_in_hand": clip_eye_in_hand,
            "reader_decoder": reader_decoder,
            "reader_prompt_pos_emb": prompt_pos_emb,
            "reader_attention_sink_tokens": attention_sink_tokens,
            "reader_current_obs_pos_emb_full": full_query_pos,
            "reader_current_obs_pos_emb_rgb": rgb_query_pos,
            "action_normalizer": action_norm,
            "bpp_unet_cond_encoder0_shape": unet_cond_shape,
            "bpp_unet_total_params": unet_total_params,
            "clip_sha256": {
                "agentview_rgb": _tensor_dict_sha256(clip_agentview),
                "eye_in_hand_rgb": _tensor_dict_sha256(clip_eye_in_hand),
            },
            "reader_sha256": _tensor_dict_sha256(reader_decoder),
        }

        with NamedTemporaryFile(dir=cache_path.parent, delete=False, suffix=".tmp") as tmp:
            tmp_path = Path(tmp.name)
        try:
            torch.save(bundle, tmp_path)
            tmp_path.replace(cache_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        summary = {
            "source": source_meta,
            "clip_agentview_params": sum(v.numel() for v in clip_agentview.values()),
            "clip_eye_in_hand_params": sum(v.numel() for v in clip_eye_in_hand.values()),
            "reader_decoder_params": sum(v.numel() for v in reader_decoder.values()),
            "reader_prompt_pos_emb_shape": list(prompt_pos_emb.shape),
            "reader_attention_sink_tokens_shape": list(attention_sink_tokens.shape),
            "reader_current_obs_pos_emb_full_shape": list(full_query_pos.shape),
            "reader_current_obs_pos_emb_rgb_shape": list(rgb_query_pos.shape),
            "reader_current_obs_rgb_indices": list(BPP_QUERY_RGB_POS_INDICES),
            "bpp_unet_cond_encoder0_shape": list(unet_cond_shape),
            "bpp_unet_total_params": unet_total_params,
            "action_normalizer_scale": action_norm["scale"].tolist(),
            "action_normalizer_offset": action_norm["offset"].tolist(),
            "clip_sha256": bundle["clip_sha256"],
            "reader_sha256": bundle["reader_sha256"],
        }
        cache_path.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
        return cache_path


def load_bpp_pretrained_weights(
    ckpt_path: Path = BPP_CKPT,
    cache_path: Path | None = None,
) -> Dict[str, Any]:
    path = extract_bpp_pretrained_weights(ckpt_path=ckpt_path, cache_path=cache_path)
    return torch.load(path, map_location="cpu", weights_only=False)
