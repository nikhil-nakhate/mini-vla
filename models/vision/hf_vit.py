from __future__ import annotations
from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F

from .registry import VisionEncoder, VisionEncoderCfg


class HFViTVisionEncoder(VisionEncoder):
    """
    HuggingFace Vision encoder wrapper (CLIP / SigLIP).

    Input:  x (B,3,H,W) float in [0,1] (or uint8 in [0,255])
    Output: (B,D) pooled CLS token, optionally projected to cfg.d_model
    """
    def __init__(
        self,
        cfg: VisionEncoderCfg,
        hf_kind: Literal["clip", "siglip"],
        default_pretrained: str,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ):
        super().__init__(cfg)
        self.cfg = cfg
        try:
            if hf_kind == "clip":
                from transformers import CLIPVisionModel
                model_id = cfg.pretrained or default_pretrained
                self.backbone = CLIPVisionModel.from_pretrained(model_id)
            elif hf_kind == "siglip":
                from transformers import SiglipVisionModel
                model_id = cfg.pretrained or default_pretrained
                self.backbone = SiglipVisionModel.from_pretrained(model_id)
            else:
                raise ValueError(f"Unsupported hf_kind={hf_kind}")
        except ImportError as e:
            raise ImportError("Install transformers: pip install transformers") from e

        hidden = int(self.backbone.config.hidden_size)
        out_dim = int(getattr(cfg, "d_model", hidden))
        self.proj = nn.Identity() if out_dim == hidden else nn.Linear(hidden, out_dim)

        # buffers for normalize (1,3,1,1)
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

        self.image_size = int(getattr(cfg, "image_size", getattr(self.backbone.config, "image_size", 224)))

        if bool(getattr(cfg, "freeze", False)):
            for p in self.backbone.parameters():
                p.requires_grad = False

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dtype == torch.uint8: x = x.float() / 255.0
            else: x = x.float()
            if x.shape[-2:] != (self.image_size, self.image_size):
                x = F.interpolate(x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

            x = (x - self.mean) / self.std
            out = self.backbone(pixel_values=x, return_dict=True).last_hidden_state
            cls = out[:, 0]
            return self.proj(cls)
