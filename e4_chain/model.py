"""Trainable E4 modules. The action decoder has a single latent input."""

import torch
from torch import nn
from torch.nn import functional as F

TOKENS, LATENT, HORIZON, TARGET = 24, 32, 192, 16
BLOCK = HORIZON // TOKENS
MODULES = {
    "A": ("action_encoder", "decoder"),
    "B": ("behavior_encoder",),
    "C": ("transfer",),
}


def transformer(width, depth):
    layer = nn.TransformerEncoderLayer(
        width,
        4,
        4 * width,
        dropout=0,
        batch_first=True,
        norm_first=True,
        activation="gelu",
    )
    return nn.TransformerEncoder(
        layer, depth, nn.LayerNorm(width), enable_nested_tensor=False
    )


class ActionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(BLOCK * 7, 128)
        self.time = nn.Parameter(torch.randn(TOKENS, 128) * 0.02)
        self.temporal = transformer(128, 3)
        self.code = nn.Sequential(nn.Linear(128, LATENT), nn.LayerNorm(LATENT))

    def forward(self, actions):
        blocks = actions.reshape(-1, TOKENS, BLOCK * 7)
        return self.code(self.temporal(self.project(blocks) + self.time))


class BehaviorEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 48), nn.GELU())
        self.frames = nn.Linear(4 * 48, 192)
        self.time = nn.Parameter(torch.randn(32, 192) * 0.02)
        self.temporal = transformer(192, 4)
        self.code = nn.Sequential(nn.Linear(192, LATENT), nn.LayerNorm(LATENT))

    def forward(self, patches):
        frames = self.frames(self.project(patches).flatten(2))
        hidden = self.temporal(frames + self.time)
        return self.code(
            F.adaptive_avg_pool1d(hidden.transpose(1, 2), TOKENS).transpose(1, 2)
        )


class Transfer(nn.Module):
    """Object tokens describe corresponding donor and query bodies at reset."""

    def __init__(self, slots):
        super().__init__()
        self.project = nn.Linear(LATENT, 96)
        self.geometry = nn.Linear(8 * 3, 96)
        self.time = nn.Parameter(torch.randn(TOKENS, 96) * 0.02)
        self.body = nn.Parameter(torch.randn(slots, 96) * 0.02)
        self.temporal = transformer(96, 2)
        self.delta = nn.Linear(96, LATENT)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def forward(self, z, donor, query):
        geometry = torch.cat((donor, query, query - donor), dim=-1)
        objects = self.geometry(geometry) + self.body
        tokens = torch.cat((self.project(z) + self.time, objects), dim=1)
        mask = torch.cat(
            (torch.zeros_like(z[:, :, 0], dtype=torch.bool), donor[:, :, -1] == 0),
            dim=1,
        )
        return self.delta(self.temporal(tokens, src_key_padding_mask=mask)[:, :TOKENS])


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(LATENT, 192)
        self.time = nn.Parameter(torch.randn(TOKENS, 192) * 0.02)
        self.temporal = transformer(192, 4)
        self.head = nn.Sequential(
            nn.Linear(192, 384), nn.GELU(), nn.Linear(384, BLOCK * TARGET)
        )

    def forward(self, z):
        return (
            self.head(self.temporal(self.project(z) + self.time))
            .reshape(-1, HORIZON, TARGET)
            .float()
        )


def osc(y):
    """Match the raw environment action order (6D OSC delta, signed gripper)."""
    return torch.cat([y[..., 8:14], y[..., 7:8]], dim=-1)


class Chain(nn.Module):
    def __init__(self, slots):
        super().__init__()
        self.action_encoder = ActionEncoder()
        self.behavior_encoder = BehaviorEncoder()
        self.transfer = Transfer(slots)
        self.decoder = Decoder()

    def configure(self, stage):
        self.stage = stage
        for name, module in self.named_children():
            active = name in MODULES[stage]
            module.requires_grad_(active).train(active)

    def predict(self, features, donor, query):
        z = self.behavior_encoder(features)
        delta = self.transfer(z, donor, query)
        return self.decoder(z + delta), z, delta

    def forward(self, target, features=None, donor=None, query=None):
        teacher = self.action_encoder(osc(target))
        if self.stage == "A":
            return self.decoder(teacher), teacher, teacher, None
        if self.stage == "B":
            z = self.behavior_encoder(features)
            return self.decoder(z), z, teacher, None
        prediction, z, delta = self.predict(features, donor, query)
        return prediction, z + delta, teacher, delta
