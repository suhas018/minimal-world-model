"""
Training loop for the World Model.

Trains:
  1. Reconstruction: "Can I recreate what I see?"
  2. KL regularization: "Is my latent space well-organized?"
"""

import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import numpy as np

from dataset import build_dataloaders
from rssm import WorldModel, reconstruction_loss, kl_loss


def train(cfg=None):
    if cfg is None:
        cfg = {
            "data": {
                "num_episodes": 100,
                "episode_length": 64,
                "seq_len": 8,
                "batch_size": 16,
                "train_split": 0.85,
                "seed": 42,
                "with_anomalies": False,
            },
            "model": {
                "embed_dim": 64,
                "action_dim": 4,
                "gru_hidden": 64,
                "latent_dim": 8,
                "latent_classes": 8,
                "encoder_channels": (16, 32, 64, 128),
                "decoder_channels": (128, 64, 32, 16),
            },
            "training": {
                "learning_rate": 3e-4,
                "total_steps": 500,
                "kl_annealing_steps": 200,
                "checkpoint_every": 250,
                "log_every": 50,
            },
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    # Data
    print("\nCollecting data...")
    train_loader, val_loader = build_dataloaders(**cfg["data"])
    print(
        "Train batches: {}  Val batches: {}".format(len(train_loader), len(val_loader))
    )

    # Model
    model = WorldModel(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print("Model parameters: {:.2f}M".format(total_params / 1e6))

    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["training"]["learning_rate"]
    )

    # ── Training state ───────────────────────────────────────────────────
    global_step = 0
    best_val_loss = float("inf")
    total_steps = cfg["training"]["total_steps"]
    kl_anneal_steps = cfg["training"]["kl_annealing_steps"]
    checkpoint_every = cfg["training"]["checkpoint_every"]
    log_every = cfg["training"]["log_every"]

    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    print("\nTraining for {} steps...".format(total_steps))
    print("-" * 60)

    # ── Main loop ────────────────────────────────────────────────────────
    while global_step < total_steps:
        model.train()

        for images, actions, next_obs in train_loader:
            if global_step >= total_steps:
                break

            images = images.to(device)  # (B, T, 1, H, W)
            actions = actions.to(device)  # (B, T)

            # KL annealing
            kl_weight = min(1.0, global_step / max(kl_anneal_steps, 1))

            # Forward
            optimizer.zero_grad()
            recons, post_logits, prior_logits = model(images, actions)

            r_loss = reconstruction_loss(recons, images)
            k_loss = kl_loss(
                post_logits,
                prior_logits,
                cfg["model"]["latent_dim"],
                cfg["model"]["latent_classes"],
                free_nats=1.0,
            )
            loss = r_loss + kl_weight * k_loss

            # Backward
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            optimizer.step()

            global_step += 1

            # Logging
            if global_step % log_every == 0:
                print(
                    "Step {:5d}/{} | loss={:.4f} | recon={:.4f} | kl={:.4f} | kl_w={:.3f}".format(
                        global_step,
                        total_steps,
                        loss.item(),
                        r_loss.item(),
                        k_loss.item(),
                        kl_weight,
                    )
                )

            # Checkpoint
            if global_step % checkpoint_every == 0:
                val_loss = evaluate(model, val_loader, device, cfg)
                print("\n  Val loss: {:.4f}".format(val_loss))

                save_checkpoint(
                    model, optimizer, global_step, val_loss, ckpt_dir / "latest.pt"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(
                        model, optimizer, global_step, val_loss, ckpt_dir / "best.pt"
                    )
                    print("  New best! val_loss={:.4f}\n".format(val_loss))

                model.train()

    print("\nTraining complete! Best val loss: {:.4f}".format(best_val_loss))


def evaluate(model, val_loader, device, cfg):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for images, actions, next_obs in val_loader:
            images = images.to(device)
            actions = actions.to(device)

            recons, post_logits, prior_logits = model(images, actions)
            r_loss = reconstruction_loss(recons, images)
            k_loss = kl_loss(
                post_logits,
                prior_logits,
                cfg["model"]["latent_dim"],
                cfg["model"]["latent_classes"],
                free_nats=1.0,
            )
            total_loss += (r_loss + k_loss).item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def save_checkpoint(model, optimizer, step, loss, path):
    torch.save(
        {
            "step": step,
            "loss": loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    print("  Saved: {}".format(path))


def load_checkpoint(path, cfg, device):
    model = WorldModel(cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["training"]["learning_rate"]
    )
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    print("Loaded checkpoint: step={}, loss={:.4f}".format(ckpt["step"], ckpt["loss"]))
    return model, optimizer, ckpt["step"]


if __name__ == "__main__":
    train()
