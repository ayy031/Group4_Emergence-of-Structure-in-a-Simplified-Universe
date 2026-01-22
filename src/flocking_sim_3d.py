import numpy as np

def run_simulation(
    N=200,
    steps=400,
    box_size=1.0,
    align=1.0,
    noise=0.05,
    R=0.15,
    speed=0.03,
    repulsion_radius=0.05,
    repulsion_strength=1.0,
    dt=1.0,
    seed=None,
    save_every=1,
    softening=1e-6,
):
    if save_every < 1:
        raise ValueError("save_every must be >= 1")

    rng = np.random.default_rng(seed)

    # positions uniform in 3D box
    pos = rng.random((N, 3)) * box_size

    # random velocity directions in 3D
    vel = rng.normal(size=(N, 3))
    vel = vel / (np.linalg.norm(vel, axis=1, keepdims=True) + 1e-12)

    history = []

    for t in range(steps):

        # pairwise displacement with periodic boundaries
        diff = pos[:, None, :] - pos[None, :, :]
        diff -= box_size * np.round(diff / box_size)
        dist = np.linalg.norm(diff, axis=2)

        # neighbors within R
        neigh = (dist > 0) & (dist < R)

        # average neighbor velocity
        count = np.sum(neigh, axis=1)
        v_sum = np.sum(vel[None, :, :] * neigh[:, :, None], axis=1)
        v_avg = v_sum / (count[:, None] + 1e-9)

        # steering
        steer = v_avg - vel
        steer[count == 0] = 0.0

        # repulsion
        rep_mask = (dist > 0) & (dist < repulsion_radius)
        rep_dir = diff / (dist[:, :, None] + softening)
        rep_weight = (repulsion_radius - dist) / repulsion_radius
        F_rep = np.sum(rep_dir * rep_weight[:, :, None] * rep_mask[:, :, None], axis=1)

        # update velocity
        vel = vel + dt * (align * steer + repulsion_strength * F_rep)
        vel = vel + noise * rng.normal(size=vel.shape)

        # normalize to constant speed
        vel = vel / (np.linalg.norm(vel, axis=1, keepdims=True) + 1e-12)
        vel = speed * vel

        # move and wrap
        pos = (pos + dt * vel) % box_size

        if (t + 1) % save_every == 0:
            history.append(pos.copy())

    return np.asarray(history)
