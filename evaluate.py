"""
Evaluation metrics for the trained world model.

Metrics:
  1. Reconstruction MSE — how well does the model recreate observations?
  2. Prediction accuracy — can it predict next frame?
  3. Imagination consistency — do imagined rollouts stay coherent?
  4. Anomaly detection rate — can it find anomalies it wasn't trained on?
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from dataset import build_dataloaders
from rssm import WorldModel
from env import FactoryEnv, Drone


def evaluate_checkpoint(cfg=None):
    if cfg is None:
        cfg = _get_default_cfg()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    # Load model
    ckpt_path = Path("checkpoints/best.pt")
    if not ckpt_path.exists():
        print("No checkpoint found. Run 'python main.py train' first.")
        return

    model = WorldModel(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("Loaded: step={}, loss={:.4f}".format(ckpt["step"], ckpt["loss"]))

    # Load data
    print("\nLoading data...")
    _, val_loader = build_dataloaders(**cfg["data"])

    # Metric 1: Reconstruction MSE
    print("\n--- Metric 1: Reconstruction MSE ---")
    recon_mse = _eval_reconstruction(model, val_loader, device)
    print("Reconstruction MSE: {:.6f}".format(recon_mse))

    # Metric 2: Imagination consistency
    print("\n--- Metric 2: Imagination Consistency ---")
    imag_consistency = _eval_imagination(model, cfg, device)
    print(
        "Imagination consistency (MSE drift over 8 steps): {:.6f}".format(
            imag_consistency
        )
    )

    # Metric 3: Anomaly detection
    print("\n--- Metric 3: Anomaly Detection ---")
    _eval_anomaly_detection(model, cfg, device)

    print("\nEvaluation complete!")


def _get_default_cfg():
    return {
        "data": {
            "num_episodes": 50,
            "episode_length": 32,
            "seq_len": 8,
            "batch_size": 16,
            "train_split": 0.85,
            "seed": 42,
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


def _eval_reconstruction(model, val_loader, device):
    """How well does the model reconstruct observations?"""
    total_mse = 0.0
    n_batches = 0

    with torch.no_grad():
        for images, actions, next_obs in val_loader:
            images = images.to(device)
            actions = actions.to(device)

            recons, _, _ = model(images, actions)
            mse = F.mse_loss(recons, images).item()
            total_mse += mse
            n_batches += 1

    return total_mse / max(n_batches, 1)


def _eval_imagination(model, cfg, device, horizon=8):
    """How consistent are imagined rollouts?"""
    model.eval()
    errors = []

    with torch.no_grad():
        for _ in range(5):
            h, z = model.rssm.initial_state(1, device)
            prev_recon = None

            for t in range(horizon):
                action = torch.randint(0, 4, (1,), device=device)
                action_onehot = F.one_hot(action, num_classes=4).float()
                h, z = model.rssm.imagine_step(h, z, action_onehot)
                feat = model.rssm.get_feature(h, z)
                recon = model.decoder(feat)

                if prev_recon is not None:
                    drift = F.mse_loss(recon, prev_recon).item()
                    errors.append(drift)

                prev_recon = recon

    return np.mean(errors) if errors else 0.0


def _eval_anomaly_detection(model, cfg, device):
    """Can the model detect anomalies it wasn't trained on?"""
    env = FactoryEnv(seed=99, with_anomalies=True, num_anomalies=5, anomaly_size=5)
    drone = Drone(rng=env.rng)
    grid, anomaly_map = env.generate()

    # Collect recon errors on normal vs anomaly positions
    normal_errors = []
    anomaly_errors = []

    obs = env.render(grid, drone.reset())
    h, z = model.rssm.initial_state(1, device)

    # Find anomaly locations and walk toward them
    anomaly_locs = np.argwhere(anomaly_map)
    if len(anomaly_locs) > 0:
        target = anomaly_locs[0]  # Walk toward first anomaly

    with torch.no_grad():
        for step_i in range(128):
            action = env.rng.randint(0, 4)

            # Bias movement toward nearest anomaly
            if len(anomaly_locs) > 0 and env.rng.random() < 0.4:
                dy = target[0] - drone.y
                dx = target[1] - drone.x
                if abs(dx) > abs(dy):
                    action = 2 if dx > 0 else 3
                else:
                    action = 1 if dy > 0 else 0

            action_tensor = (
                F.one_hot(torch.tensor([action]), num_classes=4).float().to(device)
            )
            obs_tensor = (
                torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).float().to(device)
            )

            h, z, feat, recon = model.encode_obs(obs_tensor, h, z, action_tensor)
            error = F.mse_loss(recon, obs_tensor).item()

            if drone.at_anomaly(anomaly_map):
                anomaly_errors.append(error)
            else:
                normal_errors.append(error)

            new_pos, _ = drone.step(action, grid)
            obs = env.render(grid, new_pos)

            # Update target if reached
            if len(anomaly_locs) > 0 and drone.at_anomaly(anomaly_map):
                anomaly_locs = (
                    anomaly_locs[1:] if len(anomaly_locs) > 1 else anomaly_locs
                )
                if len(anomaly_locs) > 0:
                    target = anomaly_locs[0]

    normal_mean = np.mean(normal_errors) if normal_errors else 0
    anomaly_mean = np.mean(anomaly_errors) if anomaly_errors else 0

    print(
        "Normal position errors: {:.6f} (n={})".format(normal_mean, len(normal_errors))
    )
    print(
        "Anomaly position errors: {:.6f} (n={})".format(
            anomaly_mean, len(anomaly_errors)
        )
    )

    if anomaly_mean > normal_mean:
        print("SUCCESS: Model detects anomalies (higher error on anomaly positions)")
    else:
        print(
            "NOTE: Anomaly errors not higher — may need more training or different threshold"
        )


if __name__ == "__main__":
    evaluate_checkpoint()
