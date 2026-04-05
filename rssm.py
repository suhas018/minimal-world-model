"""
RSSM — Recurrent State Space Model for 2D Grid World

Architecture:
  [64x64 obs] ─► CNN Encoder ─► embed (128)
                        │
                        ▼
  ┌─────────────────────────────────────────────┐
  │  GRU (h=128) ───► [h, z] ───► Decoder ──► [64x64] │
  │       ▲              │                      │
  │       │              ▼                      │
  │  [z_prev, action]  Posterior/Prior          │
  └─────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical


# ─────────────────────────────────────────────────────────────────────────────
# CNN Encoder: obs (B, 1, 64, 64) → embed (B, embed_dim)
# ─────────────────────────────────────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, in_channels=1, channels=(32, 64, 128, 256), embed_dim=128):
        super().__init__()
        layers = []
        c_in = in_channels
        for c_out in channels:
            layers += [
                nn.Conv2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
                nn.SiLU(),
            ]
            c_in = c_out
        self.cnn = nn.Sequential(*layers)
        # After 4 stride-2 convs on 64x64: 64→32→16→8→4 => 4x4x256=4096
        self.proj = nn.Linear(c_out * 4 * 4, embed_dim)

    def forward(self, obs):
        x = self.cnn(obs)  # (B, 256, 4, 4)
        x = x.flatten(1)  # (B, 4096)
        return self.proj(x)  # (B, embed_dim)


# ─────────────────────────────────────────────────────────────────────────────
# CNN Decoder: latent (B, latent_dim) → reconstructed obs (B, 1, 64, 64)
# ─────────────────────────────────────────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(self, latent_dim=384, channels=(256, 128, 64, 32), out_channels=1):
        super().__init__()
        self.channels = channels
        self.proj = nn.Linear(latent_dim, channels[0] * 4 * 4)
        layers = []
        c_in = channels[0]
        for c_out in channels[1:]:
            layers += [
                nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1),
                nn.SiLU(),
            ]
            c_in = c_out
        # Final layer: 8→64 needs two more upsamples
        layers += [
            nn.ConvTranspose2d(c_in, out_channels, kernel_size=4, stride=2, padding=1),
        ]
        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        x = self.proj(z)  # (B, 256*4*4)
        x = x.view(x.size(0), self.channels[0], 4, 4)  # (B, 256, 4, 4)
        return self.deconv(x)  # (B, 1, 64, 64)


# ─────────────────────────────────────────────────────────────────────────────
# RSSM Core
#   Maintains:
#     h_t — deterministic recurrent state (B, hidden_size)
#     z_t — stochastic categorical latent (B, latent_dim * latent_classes)
# ─────────────────────────────────────────────────────────────────────────────
class RSSM(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        action_dim=4,
        hidden_size=128,
        latent_dim=16,
        latent_classes=16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.latent_classes = latent_classes
        self.z_size = latent_dim * latent_classes  # 256

        # GRU input: [h_prev, z_prev, action]
        self.gru = nn.GRUCell(
            input_size=self.z_size + action_dim,
            hidden_size=hidden_size,
        )

        # Posterior: given h_t and obs embedding → z_t distribution
        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_size + embed_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, latent_dim * latent_classes),
        )

        # Prior: given h_t only → z_t distribution (used during imagination)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, latent_dim * latent_classes),
        )

    def initial_state(self, batch_size, device):
        h = torch.zeros(batch_size, self.hidden_size, device=device)
        z = torch.zeros(batch_size, self.z_size, device=device)
        return h, z

    def _logits_to_sample(self, logits):
        """Straight-through categorical sample from logits."""
        B = logits.size(0)
        logits = logits.view(B, self.latent_dim, self.latent_classes)
        dist = OneHotCategorical(logits=logits)
        sample = dist.sample()
        # Straight-through gradient
        sample = sample + (F.softmax(logits, -1) - F.softmax(logits, -1).detach())
        return sample.view(B, self.z_size)

    def observe_step(self, h, z, action, embed):
        """One step with a real observation."""
        gru_input = torch.cat([z, action], dim=-1)
        h_next = self.gru(gru_input, h)

        post_logits = self.posterior_net(torch.cat([h_next, embed], dim=-1))
        z_post = self._logits_to_sample(post_logits)

        prior_logits = self.prior_net(h_next)

        return h_next, z_post, post_logits, prior_logits

    def imagine_step(self, h, z, action):
        """One step WITHOUT observation (pure imagination)."""
        gru_input = torch.cat([z, action], dim=-1)
        h_next = self.gru(gru_input, h)
        prior_logits = self.prior_net(h_next)
        z_prior = self._logits_to_sample(prior_logits)
        return h_next, z_prior

    def get_feature(self, h, z):
        return torch.cat([h, z], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# World Model — wraps Encoder + RSSM + Decoder
# ─────────────────────────────────────────────────────────────────────────────
class WorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        m = cfg["model"]
        self.hidden_size = m["gru_hidden"]
        self.z_size = m["latent_dim"] * m["latent_classes"]
        self.feat_size = m["gru_hidden"] + self.z_size

        self.encoder = Encoder(
            in_channels=1,
            channels=tuple(m["encoder_channels"]),
            embed_dim=m["embed_dim"],
        )
        self.rssm = RSSM(
            embed_dim=m["embed_dim"],
            action_dim=m["action_dim"],
            hidden_size=m["gru_hidden"],
            latent_dim=m["latent_dim"],
            latent_classes=m["latent_classes"],
        )
        self.decoder = Decoder(
            latent_dim=self.feat_size,
            channels=tuple(m["decoder_channels"]),
            out_channels=1,
        )

    def forward(self, images, actions):
        """
        Full sequence forward pass.
        images : (B, T, 1, H, W)
        actions: (B, T)
        Returns: recons, post_logits, prior_logits
        """
        B, T = images.shape[:2]
        device = images.device

        h, z = self.rssm.initial_state(B, device)

        post_logits_list = []
        prior_logits_list = []
        recon_list = []

        for t in range(T):
            obs = images[:, t]  # (B, 1, H, W)
            action = F.one_hot(actions[:, t], num_classes=4).float()  # (B, 4)
            embed = self.encoder(obs)  # (B, embed_dim)

            h, z, post_logits, prior_logits = self.rssm.observe_step(
                h, z, action, embed
            )

            feat = self.rssm.get_feature(h, z)
            recon = self.decoder(feat)

            post_logits_list.append(post_logits)
            prior_logits_list.append(prior_logits)
            recon_list.append(recon)

        recons = torch.stack(recon_list, dim=1)  # (B, T, 1, H, W)
        post_logits = torch.stack(post_logits_list, dim=1)  # (B, T, D*C)
        prior_logits = torch.stack(prior_logits_list, dim=1)  # (B, T, D*C)

        return recons, post_logits, prior_logits

    @torch.no_grad()
    def imagine(self, h, z, actions):
        """Imagine future states given actions."""
        features = []
        for t in range(actions.shape[1]):
            h, z = self.rssm.imagine_step(h, z, actions[:, t])
            features.append(self.rssm.get_feature(h, z))
        return torch.stack(features, dim=1)

    @torch.no_grad()
    def encode_obs(self, obs, h, z, action):
        """Single step encode."""
        embed = self.encoder(obs)
        h, z, _, _ = self.rssm.observe_step(h, z, action, embed)
        feat = self.rssm.get_feature(h, z)
        recon = self.decoder(feat)
        return h, z, feat, recon


# ─────────────────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────────────────
def reconstruction_loss(recon, target):
    """MSE pixel reconstruction loss."""
    return F.mse_loss(recon, target)


def kl_loss(post_logits, prior_logits, latent_dim, latent_classes, free_nats=1.0):
    """KL divergence between posterior and prior categorical distributions."""
    B, T, _ = post_logits.shape
    post = post_logits.view(B * T, latent_dim, latent_classes)
    prior = prior_logits.view(B * T, latent_dim, latent_classes)

    post_probs = F.softmax(post, dim=-1).clamp(min=1e-8)
    prior_probs = F.softmax(prior, dim=-1).clamp(min=1e-8)

    kl = (post_probs * (post_probs.log() - prior_probs.log())).sum(-1)
    kl = kl.mean(-1)
    kl = torch.clamp(kl, min=free_nats)
    return kl.mean()


if __name__ == "__main__":
    cfg = {
        "model": {
            "embed_dim": 128,
            "action_dim": 4,
            "gru_hidden": 128,
            "latent_dim": 16,
            "latent_classes": 16,
            "encoder_channels": (32, 64, 128, 256),
            "decoder_channels": (256, 128, 64, 32),
        }
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = WorldModel(cfg).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total / 1e6:.2f}M")

    # Smoke test
    B, T, H, W = 4, 16, 64, 64
    imgs = torch.randn(B, T, 1, H, W).to(device)
    actions = torch.randint(0, 4, (B, T)).to(device)

    recons, post_logits, prior_logits = model(imgs, actions)

    print(f"Recons shape      : {recons.shape}")  # (4, 16, 1, 64, 64)
    print(f"Post logits shape : {post_logits.shape}")  # (4, 16, 256)
    print(f"Prior logits shape: {prior_logits.shape}")  # (4, 16, 256)

    recon_l = reconstruction_loss(recons, imgs)
    kl_l = kl_loss(
        post_logits,
        prior_logits,
        cfg["model"]["latent_dim"],
        cfg["model"]["latent_classes"],
    )

    print(f"Recon loss : {recon_l.item():.4f}")
    print(f"KL loss    : {kl_l.item():.4f}")

    if device.type == "cuda":
        print(f"VRAM used  : {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    print("\nrssm.py is working correctly!")
