import torch
from torch import nn
from torch.nn import functional as F


class BehaviorEncoder(nn.Module):
    """Pool local patch evidence, then contextualize 16 temporally ordered tokens."""

    def __init__(self):
        super().__init__()
        self.patch = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 72), nn.GELU())
        self.spatial = nn.Parameter(torch.randn(64, 72) * 0.02)
        self.attention = nn.Linear(72, 4)
        self.time_proj = nn.Linear(4 * 72 + 4 * 72, 96)
        self.pos = nn.Parameter(torch.randn(16, 96) * 0.02)
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                96, 4, 256, 0.05, batch_first=True, norm_first=True, activation="gelu"
            ),
            2,
            enable_nested_tensor=False,
        )
        self.latent = nn.Sequential(
            nn.LayerNorm(96), nn.Linear(96, 48), nn.LayerNorm(48)
        )

    def forward(self, features):
        x = self.patch(features.flatten(2, 3)) + self.spatial
        weights = self.attention(x).softmax(dim=2)
        attended = (weights.transpose(-1, -2) @ x).flatten(2)
        blocks = x.reshape(len(x), 32, 2, 4, 2, 4, 72).mean((3, 5)).flatten(2)
        t = (
            self.time_proj(torch.cat([attended, blocks], -1))
            .reshape(-1, 16, 2, 96)
            .mean(2)
        )
        return self.latent(self.temporal(t + self.pos))


class ActionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(48, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(128, 96, 3, padding=1),
        )
        self.head = nn.Sequential(
            nn.Conv1d(97, 96, 5, padding=2), nn.GELU(), nn.Conv1d(96, 16, 1)
        )
        self.register_buffer("phase", torch.linspace(-1, 1, 192).reshape(1, 1, -1))
        self.register_buffer("unit", torch.tensor([0.0, 0.0, 0.0, 1.0]))

    def forward(self, z):
        x = F.interpolate(
            self.temporal(z.transpose(1, 2)),
            size=192,
            mode="linear",
            align_corners=True,
        )
        y = (
            self.head(torch.cat([x, self.phase.expand(len(x), -1, -1)], 1))
            .transpose(1, 2)
            .float()
        )
        return torch.cat(
            [y[:, :, :3], F.normalize(y[:, :, 3:7] + self.unit, dim=-1), y[:, :, 7:]],
            -1,
        )


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BehaviorEncoder()
        self.decoder = ActionDecoder()

    def forward(self, feat):
        return self.decoder(self.encoder(feat))


class Transfer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(48 + 16 * 6, 64), nn.GELU(), nn.Linear(64, 48)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z, dg):
        g = dg[:, None].expand(-1, z.size(1), -1)
        return (
            z
            + self.mlp(torch.cat([z, g], -1))
            - self.mlp(torch.cat([z, torch.zeros_like(g)], -1))
        )


def loss(y, target):
    t = target
    xyz = (y[:, :, :3] - t[:, :, :3]).square().mean()
    dot = (y[:, :, 3:7] * t[:, :, 3:7]).sum(-1).abs().clamp_max(1)
    grip = t[:, :, 7].gt(0).float()
    changes = F.pad((grip[:, 1:] - grip[:, :-1]).abs(), (1, 0))
    near = F.max_pool1d(changes[:, None], 7, 1, 3)[:, 0]
    return (
        xyz
        + 3 * (1 - dot.square()).mean()
        + 0.5
        * F.binary_cross_entropy_with_logits(y[:, :, 7], grip, weight=1 + 2 * near)
        + 0.12 * F.smooth_l1_loss(y[:, :, 8:14], t[:, :, 8:14])
        + 0.12 * F.mse_loss(y[:, :, 14:], t[:, :, 14:])
    )


class RawTransfer(nn.Module):
    """Privileged control: same geometry and normalized donor poses/actions, no AE."""

    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(16 + 1 + 16 * 6, 96), nn.GELU(), nn.Linear(96, 16)
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.register_buffer("phase", torch.linspace(-1, 1, 192)[None, :, None])

    def forward(self, d, g):
        h = torch.cat([d, self.phase.expand(len(d), -1, -1)], -1)
        geom = g[:, None].expand(-1, 192, -1)
        delta = self.mlp(torch.cat([h, geom], -1)) - self.mlp(
            torch.cat([h, torch.zeros_like(geom)], -1)
        )
        y = d + delta
        return torch.cat(
            [
                y[:, :, :3],
                F.normalize(y[:, :, 3:7], dim=-1),
                d[:, :, 7:8] * 3 + delta[:, :, 7:8],
                y[:, :, 8:],
            ],
            -1,
        )
