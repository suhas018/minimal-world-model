"""
Main entry point for the minimal world model project.

Usage:
  python main.py train           # Train the world model
  python main.py evaluate        # Evaluate trained model
  python main.py plan            # Run CEM planner
  python main.py demo            # Full demo: train + plan + anomaly detect
"""

import sys
import torch
import numpy as np
from pathlib import Path


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py [train|evaluate|plan|demo]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "train":
        from trainer import train

        train()

    elif command == "evaluate":
        from evaluate import evaluate_checkpoint

        evaluate_checkpoint()

    elif command == "plan":
        from planner import CEMPlanner
        from rssm import WorldModel

        cfg = _get_default_cfg()
        device = torch.device("cpu")
        model = _load_model(cfg, device)
        planner = CEMPlanner(model, cfg, device)

        h, z = model.rssm.initial_state(1, device)
        action_dummy = torch.zeros(1, cfg["model"]["action_dim"], device=device)

        best_action, scores, best_seq = planner.plan(h, z)
        print("Best action: {}".format(best_action.cpu().numpy().round(3)))
        print(
            "Score range: [{:.4f}, {:.4f}]".format(
                scores.min().item(), scores.max().item()
            )
        )

    elif command == "demo":
        _run_demo()

    else:
        print("Unknown command: {}".format(command))
        sys.exit(1)


def _get_default_cfg():
    return {
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
        "planner": {
            "horizon": 8,
            "population": 128,
            "elite_fraction": 0.25,
            "iterations": 4,
        },
    }


def _load_model(cfg, device):
    from rssm import WorldModel

    ckpt_path = Path("checkpoints/best.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError("Run 'python main.py train' first.")
    model = WorldModel(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print("Loaded model: step={}, loss={:.4f}".format(ckpt["step"], ckpt["loss"]))
    return model


def _run_demo():
    from trainer import train
    from planner import CEMPlanner
    from env import FactoryEnv, Drone

    cfg = _get_default_cfg()
    device = torch.device("cpu")

    # Step 1: Train
    print("=" * 50)
    print("STEP 1: Training world model")
    print("=" * 50)
    train(cfg)

    # Step 2: Load model
    print("\n" + "=" * 50)
    print("STEP 2: Loading trained model")
    print("=" * 50)
    model = _load_model(cfg, device)

    # Step 3: Plan
    print("\n" + "=" * 50)
    print("STEP 3: Running CEM planner")
    print("=" * 50)
    planner = CEMPlanner(model, cfg, device)

    env = FactoryEnv(seed=99, with_anomalies=True, num_anomalies=5, anomaly_size=5)
    drone = Drone(rng=env.rng)
    grid, anomaly_map = env.generate()
    pos = drone.reset()
    obs = env.render(grid, pos)

    # Encode observation
    obs_tensor = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).float().to(device)
    action_dummy = torch.zeros(1, cfg["model"]["action_dim"], device=device)

    h, z = model.rssm.initial_state(1, device)
    with torch.no_grad():
        h, z, feat, recon = model.encode_obs(obs_tensor, h, z, action_dummy)

    # Plan
    best_action, scores, best_seq = planner.plan(h, z)
    print("Best action: {}".format(best_action.cpu().numpy().round(3)))

    # Anomaly check
    recon_error = torch.nn.functional.mse_loss(recon, obs_tensor).item()
    threshold = 0.1  # calibrated from training
    is_anomaly = recon_error > threshold
    print("Reconstruction error: {:.4f}".format(recon_error))
    print("Anomaly detected: {}".format(is_anomaly))

    # Step 4: Imagine future
    print("\n" + "=" * 50)
    print("STEP 4: Imagining future trajectory")
    print("=" * 50)
    with torch.no_grad():
        imagined = model.imagine(h, z, best_seq.unsqueeze(0))
    print("Imagined trajectory shape: {}".format(imagined.shape))
    print("Done!")


if __name__ == "__main__":
    main()
