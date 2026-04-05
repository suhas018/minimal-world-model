"""
Dataset for World Model Training

Collects trajectories by running random actions in the factory environment.
Stores sequences of (obs, action, next_obs) for training.
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from env import FactoryEnv, Drone


class TrajectoryBuffer:
    """Collects and stores trajectories for training."""

    def __init__(self, capacity: int = 100_000):
        self.capacity = capacity
        self.observations = []
        self.actions = []
        self.next_observations = []
        self.collisions = []
        self._idx = 0

    def add(self, obs: np.ndarray, action: int, next_obs: np.ndarray, collision: bool):
        if len(self.observations) < self.capacity:
            self.observations.append(obs)
            self.actions.append(action)
            self.next_observations.append(next_obs)
            self.collisions.append(collision)
        else:
            # Overwrite oldest
            idx = self._idx % self.capacity
            self.observations[idx] = obs
            self.actions[idx] = action
            self.next_observations[idx] = next_obs
            self.collisions[idx] = collision
        self._idx += 1

    def __len__(self):
        return len(self.observations)

    def sample(self, batch_size: int):
        indices = np.random.randint(0, len(self), size=batch_size)
        return (
            np.array([self.observations[i] for i in indices]),
            np.array([self.actions[i] for i in indices]),
            np.array([self.next_observations[i] for i in indices]),
            np.array([self.collisions[i] for i in indices]),
        )


def collect_trajectories(
    num_episodes: int = 100,
    episode_length: int = 64,
    seed: int = 42,
    with_anomalies: bool = True,
) -> TrajectoryBuffer:
    """
    Run random agent in factory environment to collect training data.
    """
    env = FactoryEnv(seed=seed, with_anomalies=with_anomalies)
    rng = np.random.RandomState(seed)
    buffer = TrajectoryBuffer(capacity=num_episodes * episode_length)

    for ep in range(num_episodes):
        grid, anomaly_map = env.generate()
        drone = Drone(rng=rng)
        pos = drone.reset()
        obs = env.render(grid, pos)

        for _ in range(episode_length):
            action = rng.randint(0, 4)
            new_pos, collision = drone.step(action, grid)
            next_obs = env.render(grid, new_pos)

            buffer.add(obs, action, next_obs, collision)
            obs = next_obs

        if (ep + 1) % 20 == 0:
            print("  Collected episode {}/{}".format(ep + 1, num_episodes))

    print("Total transitions: {}".format(len(buffer)))
    return buffer


class WorldModelDataset(Dataset):
    """
    Dataset for world model training.
    Returns sequences of (obs, action, next_obs) of fixed length.
    """

    def __init__(self, buffer: TrajectoryBuffer, seq_len: int = 16):
        self.seq_len = seq_len
        self.samples = []

        # Build sequential samples (non-overlapping windows)
        n = len(buffer)
        for start in range(0, n - seq_len, seq_len):
            obs_seq = []
            act_seq = []
            next_obs_seq = []

            for t in range(seq_len):
                obs_seq.append(buffer.observations[start + t])
                act_seq.append(buffer.actions[start + t])
                next_obs_seq.append(buffer.next_observations[start + t])

            self.samples.append(
                (
                    np.stack(obs_seq, axis=0),  # (T, H, W)
                    np.array(act_seq, dtype=np.int64),  # (T,)
                    np.stack(next_obs_seq, axis=0),  # (T, H, W)
                )
            )

        print("Dataset: {} sequences of length {}".format(len(self.samples), seq_len))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obs, actions, next_obs = self.samples[idx]

        # Normalize to [-0.5, 0.5] (already done in env, but ensure)
        obs = obs.astype(np.float32)
        next_obs = next_obs.astype(np.float32)

        # Add channel dimension: (T, H, W) -> (T, 1, H, W)
        obs = obs[:, None, :, :]
        next_obs = next_obs[:, None, :, :]

        return (
            torch.from_numpy(obs),
            torch.from_numpy(actions),
            torch.from_numpy(next_obs),
        )


def build_dataloaders(
    num_episodes: int = 100,
    episode_length: int = 64,
    seq_len: int = 16,
    batch_size: int = 32,
    train_split: float = 0.85,
    seed: int = 42,
    with_anomalies: bool = True,
):
    """Collect data and build train/val DataLoaders."""

    # Collect trajectories
    buffer = collect_trajectories(
        num_episodes, episode_length, seed, with_anomalies=with_anomalies
    )

    # Split into train/val
    n = len(buffer)
    cutoff = int(n * train_split)

    train_buffer = TrajectoryBuffer(capacity=cutoff)
    val_buffer = TrajectoryBuffer(capacity=n - cutoff)

    for i in range(n):
        if i < cutoff:
            train_buffer.add(
                buffer.observations[i],
                buffer.actions[i],
                buffer.next_observations[i],
                buffer.collisions[i],
            )
        else:
            val_buffer.add(
                buffer.observations[i],
                buffer.actions[i],
                buffer.next_observations[i],
                buffer.collisions[i],
            )

    train_ds = WorldModelDataset(train_buffer, seq_len)
    val_ds = WorldModelDataset(val_buffer, seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = build_dataloaders(
        num_episodes=50,
        episode_length=64,
        seq_len=16,
        batch_size=8,
    )

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")

    obs, actions, next_obs = next(iter(train_loader))
    print(f"\nBatch obs shape:      {obs.shape}")  # (B, T, 1, H, W)
    print(f"Batch actions shape:  {actions.shape}")  # (B, T)
    print(f"Batch next_obs shape: {next_obs.shape}")  # (B, T, 1, H, W)
    print(f"Obs range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"Actions: {actions.unique()}")

    print("\ndataset.py is working correctly!")
