import numpy as np


def average_nearest_neighbour_distance(positions):
    """
    Compute the average nearest-neighbour distance between agents.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 2)
        Positions of all agents in 2D space.

    Returns
    -------
    float
        Average distance to the nearest neighbour.
    """
    N = positions.shape[0]
    nearest_distances = []

    for i in range(N):
        # Compute distances from agent i to all other agents
        diff = positions[i] - positions
        distances = np.sqrt((diff ** 2).sum(axis=1))

        # Ignore distance to itself
        distances[i] = np.inf

        # Store nearest neighbour distance
        nearest_distances.append(distances.min())

    return np.mean(nearest_distances)


def largest_cluster_fraction(positions, threshold):
    """
    Compute the fraction of agents belonging to the largest cluster.

    Agents are considered connected if they are within a given distance threshold.

    Parameters
    ----------
    positions : np.ndarray, shape (N, 2)
        Positions of all agents in 2D space.
    threshold : float
        Distance threshold for cluster connectivity.

    Returns
    -------
    float
        Size of the largest cluster divided by total number of agents.
    """
    N = positions.shape[0]
    visited = np.zeros(N, dtype=bool)
    largest_cluster_size = 0

    for i in range(N):
        if visited[i]:
            continue

        # Start a new cluster
        stack = [i]
        visited[i] = True
        cluster_size = 1

        while stack:
            current = stack.pop()
            diff = positions[current] - positions
            distances = np.sqrt((diff ** 2).sum(axis=1))

            # Find neighbours within threshold
            neighbours = np.where((distances < threshold) & (~visited))[0]

            for n in neighbours:
                visited[n] = True
                stack.append(n)
                cluster_size += 1

        largest_cluster_size = max(largest_cluster_size, cluster_size)

    return largest_cluster_size / N