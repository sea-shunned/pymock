import numpy as np
import itertools
from scipy.spatial.distance import cdist
import igraph
# from numba import jit


def clusterChains(genotype, data_dict, part_clust, reduced_clust_nums):
    # Identify what base clusters the points in the new genotype are in
    new_clust_nums = [data_dict[i].base_cluster_num for i in genotype]
    print(new_clust_nums, "new_clust_nums")
    # Create a graph
    g = igraph.Graph()
    
    # Add each base cluster to the graph
    g.add_vertices(len(part_clust))

    # Add the cluster merges to the graph as edges
    g.add_edges(zip(reduced_clust_nums,new_clust_nums))
    
    # Find the connected components of this graph (the cluster chains)
    chains = list(g.components(mode="WEAK"))

    # Assign supercluster numbers to every member in each chain for easy membership check
    superclusts = np.empty(len(part_clust), dtype=int)
    for i, chain in enumerate(chains):
        superclusts[chain] = i

    return chains, superclusts

# @jit
def objCNN(chains, superclusts, cnn_pairs, conn_array, max_conn):
    # Saves time for a single cluster solutions
    if len(chains) == 1:
        conn_score = 0

    else:
        conn_score = max_conn

        # Loop over the actual pairs that can contribute to the CNN
        for pair in cnn_pairs:
            # See if they are part of the same supercluster i.e. have they merged
            if superclusts[pair[0]] == superclusts[pair[1]]:
                conn_score -= conn_array[pair[0], pair[1]]

    return conn_score

# @profile
def objVAR(chains, part_clust, base_members, base_centres, superclusts):
    members = np.zeros((len(chains), 1))
    centres = np.zeros((len(chains), part_clust[0].centroid.squeeze().shape[0]))
    variances = np.sum([obj.intraclust_var for obj in part_clust.values()])
    wcss_vec = np.zeros((len(chains),1))

    # Loop over all the chains/superclusters
    for superclust, chain in enumerate(chains):
        centres[superclust, :] = np.sum(np.multiply(base_centres[chain], base_members[chain]), axis=0)
        members[superclust] = np.sum(base_members[chain])

        # wcss_vec[superclust] = members[superclust] * np.dot(base_centres[chain]-centres[superclust],base_centres[chain]-centres[superclust])
        sub_res = base_centres[chain]-centres[superclust]		
        # wcss_vec[superclust] = np.sum(np.einsum('ij,ij->i',sub_res,sub_res) * base_members[chain].squeeze())

        wcss_vec[superclust] = np.sum(np.multiply(base_members[chain], np.einsum('ij,ij->i', sub_res, sub_res)))

    centres = np.divide(centres, members)

    wcss = np.sum([part_clust[index].num_members * np.dot(part_clust[index].centroid.squeeze() - centres[value], part_clust[index].centroid.squeeze() - centres[value]) for index, value in enumerate(superclusts)])

    # print(wcss, np.sum(wcss_vec), wcss_vec)

    return wcss + variances

# @profile
# def objVAR3(chains, part_clust, base_members, base_centres, superclusts):
# 	members = np.zeros((len(chains),1))
# 	centres = np.zeros((len(chains),part_clust[0].centroid.squeeze().shape[0]))
# 	variances = np.sum([obj.intraclust_var for obj in part_clust.values()])

# 	wcss_vec = np.zeros((len(chains),1))

# 	for superclust, chain in enumerate(chains):
# 		centres[superclust,:] = np.sum(np.multiply(base_centres[chain],base_members[chain]),axis=0)
# 		members[superclust] = np.sum(base_members[chain])

# 	centres = np.divide(centres, members)

# 	for superclust, chain in enumerate(chains):

# 		sub_res = base_centres[chain]-centres[superclust]
# 		# print(sub_res.shape, base_members[chain].shape, base_members[chain].squeeze().shape)
# 		# if base_members[chain].shape[0]==1:
# 		# 	print(base_members[chain], base_members[chain].squeeze())
# 		wcss_vec[superclust] = np.einsum('i,i->',np.einsum('ij,ij->i',sub_res,sub_res),base_members[chain].squeeze())

# 	return np.sum(wcss_vec) + variances


# @profile
def evalMOCK(genotype, part_clust, reduced_clust_nums, conn_array, max_conn, num_examples, data_dict, cnn_pairs, base_members, base_centres):
    # Not really necessary but just in case
    if len(genotype) != len(reduced_clust_nums):
        raise ValueError("The genotype being evaluated is not the same length as the reduced, partial genotype")

    chains, superclusts = clusterChains(genotype, data_dict, part_clust, reduced_clust_nums)

    # if len(chains) == 1:
        # CNN2 = 0
        # VAR3 = # create func for single cluster?

    CNN = objCNN(chains, superclusts, cnn_pairs, conn_array, max_conn)
    VAR = objVAR(chains, part_clust, base_members, base_centres, superclusts)
    
    # VAR2 = objVAR3(chains, part_clust, base_members, base_centres, superclusts)
    # print(np.isclose(VAR,VAR2), VAR, VAR2)

    # Edge cases with large datasets, as single cluster solutions get given a 0 score
    if CNN < 0.00001: # Value that Garza/Handl uses, presumably to smooth small values
        CNN = 0

    # Divide by number of examples as per the paper and Mario's code
    return np.sum(VAR)/num_examples, CNN