"""V-JEPA 2.1 ViT-B encoder loading (frozen, official checkpoint).

Reuses the official facebookresearch/vjepa2 code vendored under
third_party/vjepa2. The hub helper is bypassed because src/hub/backbones.py
ships a localhost VJEPA_BASE_URL; we follow the same recipe it encodes
(vit_base arch, ema_encoder key) and load the .pt directly.
"""

from __future__ import annotations

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parent
VJEPA2 = ROOT / "third_party" / "vjepa2"
CKPT = ROOT / "checkpoints" / "vjepa2_1_vitb_dist_vitG_384.pt"

IMG_SIZE = 384
TUBELET = 2
PATCH = 16
EMBED = 768
MEAN = torch.tensor([0.485, 0.456, 0.406]) * 255.0
STD = torch.tensor([0.229, 0.224, 0.225]) * 255.0

for p in (VJEPA2, VJEPA2 / "app"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _clean(sd):
    out = {}
    for k, v in sd.items():
        out[k.replace("module.", "").replace("backbone.", "")] = v
    return out


def load_encoder(device="cuda"):
    """Official vjepa2_1_vit_base_384 recipe from src/hub/backbones.py."""
    from app.vjepa_2_1.models import vision_transformer as vit

    enc = vit.vit_base(
        patch_size=PATCH,
        img_size=(IMG_SIZE, IMG_SIZE),
        num_frames=64,
        tubelet_size=TUBELET,
        use_sdpa=True,
        use_SiLU=False,
        wide_SiLU=True,
        uniform_power=False,
        use_rope=True,
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    enc.load_state_dict(_clean(sd["ema_encoder"]), strict=True)
    enc.eval().to(device)
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def preprocess_batch(frames_u8, device):
    """numpy (T,H,W,3) uint8 -> (1,3,T,384,384) normalized, on device."""
    x = torch.from_numpy(frames_u8).permute(0, 3, 1, 2).float()    # T 3 H W
    x = torch.nn.functional.interpolate(
        x, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False)
    x = x.permute(1, 0, 2, 3).unsqueeze(0)                         # 1 3 T H W
    x = (x - MEAN.view(1, 3, 1, 1, 1)) / STD.view(1, 3, 1, 1, 1)
    return x.to(device)
