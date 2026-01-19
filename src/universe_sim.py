import numpy as np
from typing import Optional


def initialize_particles(N: int, box_size: float, rng: np.random.Generator) -> np.ndarray:
    """Uniform random initial positions in a 2D periodic box."""
    return rng.random((N, 2)) * box_size


def step(
    positions: np.ndarray,
    rng: np.random.Generator,
    attraction: float = 0.01,
    repulsion: float = 0.01,
    noise: float = 0.01,
    box_size: float = 1.0,
    dt: float = 1.0,
    repulsion_radius: float = 0.05,
) -> np.ndarray:
    """
    One simple update:
    - local attraction within a finite interaction range
    - short-range repulsion prevents overlap
    - Gaussian noise perturbs positions
    - periodic boundary conditions via modulo
    """
    # pairwise displacement with periodic boundary conditions
    diffs = positions[:, None, :] - positions[None, :, :]
    diffs -= box_size * np.round(diffs / box_size)
    dists = np.linalg.norm(diffs, axis=2)
    
    softening = 1e-3
    interaction_range = 0.6 
    
  # --- local attraction towards local centre of mass ---
    att_mask = (dists > 0) & (dists < interaction_range)
    count = np.sum(att_mask, axis=1)

    neighbor_sum = np.sum(positions[None, :, :] * att_mask[:, :, None], axis=1)
    neighbor_mean = neighbor_sum / (count[:, None] + 1e-9)

    F_att = neighbor_mean - positions
   
    # short-range repulsion (closer -> stronger)
    rep_mask = (dists > 0) & (dists < repulsion_radius)
    rep_dir = diffs / (dists[:, :, None] + softening)
    rep_weight = (repulsion_radius - dists)
    F_rep = np.sum(rep_dir * rep_weight[:, :, None] * rep_mask[:, :, None], axis=1)

    positions = positions + dt * (attraction * F_att + repulsion * F_rep)
    positions = positions + noise * rng.normal(size=positions.shape)
    return positions % box_size


    # noise
    positions = positions + noise * rng.normal(size=positions.shape)

    # periodic boundary
    return positions % box_size


def run_simulation(
    N: int = 200,
    steps: int = 100,
    box_size: float = 1.0,
    attraction: float = 0.01,
    repulsion: float = 0.02,
    noise: float = 0.01,
    dt: float = 1.0,
    seed: Optional[int] = None,
    save_every: int = 1,
) -> np.ndarray:
    """
    Run the simulation and return an array of saved configurations.

    seed:
        Makes runs reproducible.
    save_every:
        Save positions every k steps (k=1 means save every step).
        If you only care about the final configuration, use save_every=steps.
    """
    if save_every < 1:
        raise ValueError("save_every must be >= 1")

    rng = np.random.default_rng(seed)
    positions = initialize_particles(N, box_size, rng)

    history = []
    for t in range(steps):
        positions = step(
            positions,
            rng=rng,
            attraction=attraction,
            repulsion=repulsion,
            noise=noise,
            box_size=box_size,
            dt=dt
        )
        if (t + 1) % save_every == 0:
            history.append(positions.copy())

    return np.asarray(history)