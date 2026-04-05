"""
2D Grid-World Factory Environment

Factory floor = 64x64 grid:
  0 = empty space
  1 = wall/machine (obstacle)
  2 = anomaly (leak, fire, defect)

Drone moves in 4 directions: N, S, E, W
Observation = 64x64 grayscale image (with noise)
"""

import numpy as np
from typing import Tuple, Optional


class FactoryEnv:
    """Generates random factory floors with walls and anomalies."""

    def __init__(
        self,
        grid_size: int = 64,
        wall_density: float = 0.15,
        num_anomalies: int = 2,
        anomaly_size: int = 3,
        with_anomalies: bool = True,
        seed: Optional[int] = None,
    ):
        self.grid_size = grid_size
        self.wall_density = wall_density
        self.num_anomalies = num_anomalies
        self.anomaly_size = anomaly_size
        self.with_anomalies = with_anomalies
        self.rng = np.random.RandomState(seed)

    def generate(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            grid: (H, W) int array — 0=empty, 1=wall, 2=anomaly
            anomaly_map: (H, W) bool array — True where anomalies are
        """
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)

        # Scatter walls
        wall_mask = self.rng.random(grid.shape) < self.wall_density
        grid[wall_mask] = 1

        # Place anomalies (clusters of 3x3 or 5x5)
        anomaly_map = np.zeros(grid.shape, dtype=bool)
        if self.with_anomalies:
            for _ in range(self.num_anomalies):
                cx = self.rng.randint(5, self.grid_size - 5)
                cy = self.rng.randint(5, self.grid_size - 5)
                half = self.anomaly_size // 2
                grid[cy - half : cy + half + 1, cx - half : cx + half + 1] = 2
                anomaly_map[cy - half : cy + half + 1, cx - half : cx + half + 1] = True

        return grid, anomaly_map

    def render(self, grid: np.ndarray, drone_pos: Tuple[int, int]) -> np.ndarray:
        """
        Render a visual observation from the drone's perspective.
        Returns: (H, W) float32 in [-0.5, 0.5] with noise.
        """
        img = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Empty = dark gray, walls = white, anomalies = bright red-ish
        img[grid == 0] = 0.1
        img[grid == 1] = 0.9
        img[grid == 2] = 0.7

        # Add sensor noise
        img += self.rng.normal(0, 0.02, img.shape)

        # Drone position marker
        dx, dy = drone_pos
        img[dy - 1 : dy + 2, dx - 1 : dx + 2] = 0.5

        # Normalize to [-0.5, 0.5]
        img = img - 0.5
        return img


class Drone:
    """Simple drone that moves on the grid."""

    ACTIONS = {
        0: (0, -1),  # North
        1: (0, 1),  # South
        2: (1, 0),  # East
        3: (-1, 0),  # West
    }

    def __init__(
        self, grid_size: int = 64, rng: Optional[np.random.RandomState] = None
    ):
        self.grid_size = grid_size
        self.rng = rng or np.random.RandomState()
        self.reset()

    def reset(self) -> Tuple[int, int]:
        """Place drone at random empty position."""
        self.x = self.rng.randint(5, self.grid_size - 5)
        self.y = self.rng.randint(5, self.grid_size - 5)
        return self.x, self.y

    def step(self, action: int, grid: np.ndarray) -> Tuple[Tuple[int, int], bool]:
        """
        Move drone. Returns (new_pos, collision).
        Collision = True if wall blocks movement.
        """
        dx, dy = self.ACTIONS[action]
        nx = np.clip(self.x + dx, 0, self.grid_size - 1)
        ny = np.clip(self.y + dy, 0, self.grid_size - 1)

        collision = grid[ny, nx] == 1
        if not collision:
            self.x, self.y = nx, ny

        return (self.x, self.y), collision

    def at_anomaly(self, anomaly_map: np.ndarray) -> bool:
        """Check if drone is currently on an anomaly."""
        return anomaly_map[self.y, self.x]


if __name__ == "__main__":
    env = FactoryEnv(seed=42)
    drone = Drone(rng=env.rng)

    grid, anomaly_map = env.generate()
    pos = drone.reset()
    obs = env.render(grid, pos)

    print(f"Grid shape: {grid.shape}")
    print(f"Anomalies: {anomaly_map.sum()} pixels")
    print(f"Drone at: {pos}")
    print(f"Obs range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"On anomaly: {drone.at_anomaly(anomaly_map)}")

    # Simulate a few steps
    for i in range(5):
        action = env.rng.randint(0, 4)
        new_pos, collision = drone.step(action, grid)
        on_anomaly = drone.at_anomaly(anomaly_map)
        print(
            f"  Step {i}: action={action}, pos={new_pos}, collision={collision}, anomaly={on_anomaly}"
        )

    print("\nenv.py is working correctly!")
