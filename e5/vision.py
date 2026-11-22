"""Frozen BPP CLIP ViT-B/16 dual-camera visual encoder with deterministic preprocessing."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from . import BPP_CKPT, CAMERAS, CENTER_CROP_RATIO, EMBED_DIM, IMAGE_SIZE
from .bpp_compat import load_bpp_pretrained_weights

CLIP_MODEL_NAME = "vit_base_patch16_clip_224.openai"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
CROP_SIZE = int(IMAGE_SIZE * CENTER_CROP_RATIO)  # 212, matches BPP ImageAugmentation


def preprocess_raw_uint8_hwc(
    images_nhwc: np.ndarray | torch.Tensor,
    convention: str = "opengl",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Exact deterministic BPP eval image preprocessing:
    1. OpenGL vertical flip (matches `_receding_rgb_numpy_thwc_to_float_chw` & `upright_pair`)
    2. uint8 [0, 255] -> float32 [0, 1] NCHW
    3. Bilinear resize to 224x224 (`align_corners=False`, matches dataset pipeline)
    4. CenterCrop(212) -> Resize(224, antialias=True) (matches BPP `eval_image_transforms`)
    5. OpenAI CLIP channel normalization
    """
    if isinstance(images_nhwc, np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(images_nhwc))
    else:
        x = images_nhwc
    if device is not None:
        x = x.to(device=device, non_blocking=True)
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"Expected (N, H, W, 3) images, got {tuple(x.shape)}")
    if convention == "opengl":
        x = torch.flip(x, dims=[1])
    elif convention != "opencv":
        raise ValueError(f"Unknown camera convention: {convention}")
    x = x.permute(0, 3, 1, 2).contiguous().to(dtype=torch.float32)
    if images_nhwc.dtype == np.uint8 or (torch.is_tensor(images_nhwc) and images_nhwc.dtype == torch.uint8):
        x = x / 255.0
    if x.shape[2:] != (IMAGE_SIZE, IMAGE_SIZE):
        x = F.interpolate(
            x, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=False
        )
    return preprocess_float_224_nchw(x)


def preprocess_float_224_nchw(x: torch.Tensor) -> torch.Tensor:
    """Apply BPP's deterministic eval_image_transforms + CLIP normalization on upright
    float32 (N, 3, 224, 224) images in [0, 1]."""
    if x.ndim != 4 or x.shape[1:] != (3, IMAGE_SIZE, IMAGE_SIZE):
        raise ValueError(f"Expected (N, 3, 224, 224), got {tuple(x.shape)}")
    x = TF.center_crop(x, [CROP_SIZE, CROP_SIZE])
    x = TF.resize(x, [IMAGE_SIZE, IMAGE_SIZE], antialias=True)
    mean = x.new_tensor(CLIP_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(CLIP_STD).view(1, 3, 1, 1)
    return (x - mean) / std


class BPPVisualEncoder(nn.Module):
    """Completely frozen dual-camera CLIP ViT-B/16 encoder initialized strictly from BPP's
    trained LIBERO-Gen Goal Chain checkpoint (`share_rgb_model=False`)."""

    def __init__(
        self,
        pretrained_bundle: Optional[Dict] = None,
        ckpt_path: Path = BPP_CKPT,
        load_weights: bool = True,
    ):
        super().__init__()
        self.key_model_map = nn.ModuleDict()
        for cam in CAMERAS:
            backbone = timm.create_model(
                model_name=CLIP_MODEL_NAME,
                pretrained=False,
                global_pool="",
                num_classes=0,
            )
            self.key_model_map[cam] = backbone

        if load_weights:
            bundle = (
                pretrained_bundle
                if pretrained_bundle is not None
                else load_bpp_pretrained_weights(ckpt_path=ckpt_path)
            )
            self.key_model_map["agentview_rgb"].load_state_dict(
                bundle["clip_agentview"], strict=True
            )
            self.key_model_map["eye_in_hand_rgb"].load_state_dict(
                bundle["clip_eye_in_hand"], strict=True
            )
            self.weight_sha256 = dict(bundle["clip_sha256"])
        else:
            self.weight_sha256 = {}

        # Completely freeze both backbones permanently.
        for param in self.parameters():
            param.requires_grad_(False)
        super().train(False)

    def train(self, mode: bool = True) -> "BPPVisualEncoder":
        # Keep frozen CLIP in eval mode even if a parent module calls .train(True).
        super().train(False)
        return self

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @torch.inference_mode()
    def encode_camera_preprocessed(
        self,
        camera: str,
        x_norm: torch.Tensor,
        batch_size: int = 512,
        autocast_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Encode already preprocessed (N, 3, 224, 224) tensors -> CLS tokens (N, 768) fp32."""
        model = self.key_model_map[camera]
        dev = self.device
        out = torch.empty((x_norm.shape[0], EMBED_DIM), dtype=torch.float32, device=dev)
        amp_ctx = (
            torch.autocast(device_type=dev.type, dtype=autocast_dtype)
            if (autocast_dtype is not None and dev.type == "cuda")
            else torch.autocast(device_type=dev.type, enabled=False)
        )
        for start in range(0, x_norm.shape[0], batch_size):
            end = min(start + batch_size, x_norm.shape[0])
            chunk = x_norm[start:end].to(device=dev, non_blocking=True)
            with amp_ctx:
                tokens = model(chunk)
            out[start:end] = tokens[:, 0, :].float()
        return out

    @torch.inference_mode()
    def encode_dual_camera_uint8(
        self,
        agentview_nhwc: np.ndarray | torch.Tensor,
        eye_in_hand_nhwc: np.ndarray | torch.Tensor,
        convention: str = "opengl",
        batch_size: int = 512,
        autocast_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Encode raw uint8 (N, H, W, 3) frames from both cameras -> (N, 2, 768) float32."""
        n = len(agentview_nhwc)
        if len(eye_in_hand_nhwc) != n:
            raise ValueError(
                f"Mismatched camera frame counts: {n} vs {len(eye_in_hand_nhwc)}"
            )
        dev = self.device
        feats = torch.empty((n, 2, EMBED_DIM), dtype=torch.float32, device=dev)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            av = preprocess_raw_uint8_hwc(
                agentview_nhwc[start:end], convention=convention, device=dev
            )
            eh = preprocess_raw_uint8_hwc(
                eye_in_hand_nhwc[start:end], convention=convention, device=dev
            )
            feats[start:end, 0] = self.encode_camera_preprocessed(
                "agentview_rgb", av, batch_size=batch_size, autocast_dtype=autocast_dtype
            )
            feats[start:end, 1] = self.encode_camera_preprocessed(
                "eye_in_hand_rgb", eh, batch_size=batch_size, autocast_dtype=autocast_dtype
            )
        return feats

    @torch.inference_mode()
    def encode_dual_camera_float_224(
        self,
        agentview_nchw: torch.Tensor,
        eye_in_hand_nchw: torch.Tensor,
        batch_size: int = 512,
        autocast_dtype: Optional[torch.dtype] = None,
    ) -> torch.Tensor:
        """Online interface for upright float32 [0, 1] (N, 3, 224, 224) images -> (N, 2, 768)."""
        dev = self.device
        av = preprocess_float_224_nchw(agentview_nchw.to(device=dev, dtype=torch.float32))
        eh = preprocess_float_224_nchw(eye_in_hand_nchw.to(device=dev, dtype=torch.float32))
        av_f = self.encode_camera_preprocessed(
            "agentview_rgb", av, batch_size=batch_size, autocast_dtype=autocast_dtype
        )
        eh_f = self.encode_camera_preprocessed(
            "eye_in_hand_rgb", eh, batch_size=batch_size, autocast_dtype=autocast_dtype
        )
        return torch.stack([av_f, eh_f], dim=1)

    def parameter_count(self) -> Tuple[int, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
