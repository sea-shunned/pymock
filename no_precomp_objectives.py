from individual import init_test
import numpy as np
import igraph
from precompute import compute_dists, L_nn
import individual
from initialisation import init_uniformly_distributed_population
from classes import MOCKGenotype


def calc_base_clusters(genotype):
    """ Calculates clusters in a genotype.
    genotpye: a list as a full genotype.
    Returns: a list of clusters.
    Taken from: Classes 338"""
    # Create a graph
    g = igraph.Graph()
    # Add the nodes
    g.add_vertices(len(genotype))
    # Add the links
    g.add_edges(zip(range(len(genotype)), genotype))

    return list(g.components(mode="WEAK"))


def CNN(nn, clusters):
    """Add a penalty for every nn that is not in the same cluster.
    nn: list of indices of L nearest neighbours.
    clusters: list of indices of clusters.
    Returns: CNN score"""
    # Speed up evaluation when number of clusters is 1
    if len(clusters) == 1:
        return 0

    score = 0
    for cluster in clusters:
        for node in cluster:
            for l, neighbour in enumerate(nn[node]):
                # If node's neighbour is not in the same cluster, add penalty
                if neighbour not in cluster:
                    score += 1/(l+2)

    return score


def VAR(data, clusters):
    """VAR score as defined in Garza (2018)
    data: np array.
    clusters: list of indices of clusters"""
    # Speed up evaluation if every cluster is composed by only one point
    if len(clusters) == data.shape[0]:
        return 0

    score = 0
    for cluster_idx in clusters:
        cluster = data[cluster_idx]
        centroid = np.mean(cluster, axis=0)
        dissimilarities = compute_dists(cluster, centroid.reshape(1, -1))
        score += np.square(dissimilarities).sum()

    return score/data.shape[0]


def evaluate_mock(ind, di_index, data, mst, Lnn):
    # Resemble full individual genotype
    full_genotype = individual.rebuild_ind_genotype(genotype=ind, di_index=di_index, mst=mst)

    # Calculate clusters
    clusters = calc_base_clusters(full_genotype)

    cnn = CNN(Lnn, clusters)
    var = VAR(data, clusters)
    if cnn < 0.00001:  # Value that Garza/Handl uses, presumably to smooth small values
        cnn = 0
    return var, cnn


# To test
if __name__ == '__main__':
    data = np.array([0.5, 0.5, 0.65, 0.55, 0.6, 0.6, 0, 0, -0.1, -0.05, 0.05, 0]).reshape((6, 2))
    distances = compute_dists(data, data)
    mst, toolbox = init_test(n=6, nclusters=3, min_delta=60, max_delta=99)
    ind = toolbox.individual()
    L = 2
    nn = L_nn(distances, L)
    clusters = calc_base_clusters(mst)

    MOCKGenotype.n_links = len(mst)
    MOCKGenotype.mst_genotype = mst

    print(ind)
    print(mst)
    print(data)
    print(distances)
    print(distances.argsort())
    print(nn)
    print(clusters)
    print(evaluate_mock(ind, ind, data, mst, nn))

    pop = init_uniformly_distributed_population(10, 5, 0, 100, distances.argsort(), 5, toolbox.individual, 0)
    print(pop)
    print([i.delta for i in pop])
