import numpy as np


def nearest_neighbor_distance(positions, box_size):
    """
    Compute average nearest-neighbour distance under periodic boundary conditions.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 2)
        Particle positions.
    box_size : float
        Size of the simulation box (assumed square).

    Returns
    -------
    float
        Mean nearest-neighbour distance.
    """
    N = positions.shape[0]
    distances = []

    for i in range(N):
        diff = positions - positions[i]
        diff -= box_size * np.round(diff / box_size)  # periodic BC
        dists = np.linalg.norm(diff, axis=1)
        dists = dists[dists > 0]  # remove self-distance
        distances.append(np.min(dists))

    return np.mean(distances)


def largest_cluster_fraction(positions, eps, box_size):
    """
    Compute fraction of particles in the largest cluster.
    Simple distance-based clustering.

    Parameters
    ----------
    positions : np.ndarray
    eps : float
        Distance threshold to define neighbours.
    box_size : float

    Returns
    -------
    float
        Size of largest cluster divided by N.
    """
    N = positions.shape[0]
    visited = np.zeros(N, dtype=bool)
    clusters = []

    for i in range(N):
        if visited[i]:
            continue

        stack = [i]
        cluster = []

        while stack:
            j = stack.pop()
            if visited[j]:
                continue
            visited[j] = True
            cluster.append(j)

            diff = positions - positions[j]
            diff -= box_size * np.round(diff / box_size)
            dists = np.linalg.norm(diff, axis=1)

            neighbors = np.where(dists < eps)[0]
            for n in neighbors:
                if not visited[n]:
                    stack.append(n)

        clusters.append(cluster)

    largest = max(len(c) for c in clusters)
    return largest / N


def density_variance_grid(positions: np.ndarray, box_size: float, bins: int = 20, normalized: bool = True) -> float:
    H, _, _ = np.histogram2d(
        positions[:, 0], positions[:, 1],
        bins=bins,
        range=[[0, box_size], [0, box_size]],
    )
    mean = np.mean(H)
    var = np.var(H)
    if normalized:
        return var / (mean**2 + 1e-12)
    return float(var)

def number_of_clusters(positions: np.ndarray, eps: float, box_size: float, min_size: int = 1) -> int:
    """
    Count number of clusters under a simple distance-threshold connectivity rule (PBC).

    Two particles are connected if their distance (under periodic BC) is < eps.
    A cluster is a connected component in this graph.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 2)
    eps : float
        Neighbour threshold distance.
    box_size : float
    min_size : int
        Only count clusters with size >= min_size (useful to ignore singletons).

    Returns
    -------
    int
        Number of clusters.
    """
    N = positions.shape[0]
    visited = np.zeros(N, dtype=bool)
    n_clusters = 0

    for i in range(N):
        if visited[i]:
            continue

        stack = [i]
        cluster_size = 0

        while stack:
            j = stack.pop()
            if visited[j]:
                continue
            visited[j] = True
            cluster_size += 1

            diff = positions - positions[j]
            diff -= box_size * np.round(diff / box_size)
            dists = np.linalg.norm(diff, axis=1)

            neighbors = np.where(dists < eps)[0]
            for n in neighbors:
                if not visited[n]:
                    stack.append(n)

        if cluster_size >= min_size:
            n_clusters += 1

    return n_clusters