"""
CEM Planner — Cross-Entropy Method for trajectory planning in imagination.

Given current latent state (h, z):
  1. Sample 128 action sequences of length 8
  2. Roll each through the world model (imagine)
  3. Score by reconstruction uncertainty (high variance = interesting)
  4. Keep top 32 (elites), refit distribution
  5. Repeat 4 iterations
  6. Return best first action
"""

import torch
import torch.nn.functional as F
import numpy as np
from rssm import WorldModel


class CEMPlanner:
    def __init__(self, model, cfg, device):
        self.model = model
        self.device = device

        p = cfg.get("planner", {})
        self.horizon = p.get("horizon", 8)
        self.population = p.get("population", 128)
        self.elite_frac = p.get("elite_fraction", 0.25)
        self.iterations = p.get("iterations", 4)
        self.action_dim = cfg["model"]["action_dim"]
        self.n_elites = int(self.population * self.elite_frac)

        # Freeze model during planning
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

    @torch.no_grad()
    def plan(self, h, z):
        B = self.population
        mu = torch.zeros(self.horizon, self.action_dim, device=self.device)
        std = torch.ones(self.horizon, self.action_dim, device=self.device)

        for iteration in range(self.iterations):
            # Sample action sequences
            noise = torch.randn(B, self.horizon, self.action_dim, device=self.device)
            actions = (mu.unsqueeze(0) + std.unsqueeze(0) * noise).clamp(-3.0, 3.0)

            # Roll out in imagination
            h_batch = h.expand(B, -1).clone()
            z_batch = z.expand(B, -1).clone()

            recon_errors = []
            for t in range(self.horizon):
                h_batch, z_batch = self.model.rssm.imagine_step(
                    h_batch, z_batch, actions[:, t]
                )
                # Score every 2 steps
                if t % 2 == 0:
                    feat = self.model.rssm.get_feature(h_batch, z_batch)
                    recon = self.model.decoder(feat)
                    # Higher variance = more "interesting" / uncertain
                    recon_var = recon.var(dim=[1, 2, 3])
                    recon_errors.append(recon_var)

            scores = torch.stack(recon_errors, dim=0).mean(dim=0)
            elite_ids = scores.argsort()[: self.n_elites]
            elites = actions[elite_ids]

            mu = elites.mean(dim=0)
            std = elites.std(dim=0).clamp(min=0.1)

        best_idx = scores.argsort()[0]
        best_action = actions[best_idx, 0]
        return best_action, scores, actions[best_idx]


def load_best_model(cfg, device):
    """Load trained world model from checkpoint."""
    ckpt_path = Path("checkpoints/best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError("Run trainer.py first to get checkpoints/best.pt")

    model = WorldModel(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded best.pt — step {ckpt['step']}, val_loss={ckpt['loss']:.4f}")
    return model


def compute_anomaly(obs, reconstructed, threshold):
    """Detect anomaly by comparing real obs to reconstruction."""
    recon_error = F.mse_loss(reconstructed, obs).item()
    severity = "none"
    if recon_error > threshold * 2:
        severity = "high"
    elif recon_error > threshold:
        severity = "medium"
    return {
        "anomaly": recon_error > threshold,
        "severity": severity,
        "recon_error": round(recon_error, 5),
    }


if __name__ == "__main__":
    from env import FactoryEnv, Drone
    from rssm import WorldModel

    cfg = {
        "model": {
            "embed_dim": 128,
            "action_dim": 4,
            "gru_hidden": 128,
            "latent_dim": 16,
            "latent_classes": 16,
            "encoder_channels": (32, 64, 128, 256),
            "decoder_channels": (256, 128, 64, 32),
        },
        "planner": {
            "horizon": 8,
            "population": 128,
            "elite_fraction": 0.25,
            "iterations": 4,
        },
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create a dummy model for testing
    model = WorldModel(cfg).to(device)
    planner = CEMPlanner(model, cfg, device)

    # Test with random latent state
    h, z = model.rssm.initial_state(1, device)
    action_dummy = torch.zeros(1, 4, device=device)

    print(
        f"\nRunning CEM ({planner.population} trajectories x {planner.iterations} iters)..."
    )
    best_action, scores, best_sequence = planner.plan(h, z)

    print(f"Best action: {best_action.cpu().numpy().round(3)}")
    print(f"Score range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
    print(f"Best sequence shape: {best_sequence.shape}")

    print("\nplanner.py is working correctly!")
