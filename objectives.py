import numpy as np
import igraph


def cluster_chains(genotype, data_dict, comp_dict, reduced_clust_nums):
    # Identify what base clusters the points in the new genotype are in
    new_clust_nums = [data_dict[i].base_cluster_num for i in genotype]
    # Create a graph
    g = igraph.Graph()
    # Add each base cluster to the graph
    g.add_vertices(len(comp_dict))
    # Add the cluster merges to the graph as edges
    g.add_edges(zip(reduced_clust_nums, new_clust_nums))
    # Find the connected components of this graph (the cluster chains)
    chains = list(g.components(mode="WEAK"))

    ### Add the number of clusters encoded by this individual
    ### Either add to the Genotype class when implemented or to DEAP creator
    # try:
    #     genotype.num_clusts = len(chains)
    # except AttributeError:
    #     pass

    # Assign supercluster numbers to every member in each chain for easy membership check
    superclusts = np.empty(len(comp_dict), dtype=int)
    for i, chain in enumerate(chains):
        superclusts[chain] = i
    return chains, superclusts


def objCNN(chains, superclusts, cnn_pairs, cnn_array, max_cnn):
    # Saves time for a single cluster solutions
    if len(chains) == 1:
        conn_score = 0
    else:
        conn_score = max_cnn
        # Loop over the actual pairs that can contribute to the CNN
        for pair in cnn_pairs:
            # See if they are part of the same supercluster i.e. have they merged
            if superclusts[pair[0]] == superclusts[pair[1]]:
                conn_score -= cnn_array[pair[0], pair[1]]
    return conn_score


# @profile
def objVAR(chains, comp_dict, base_members, base_centres, superclusts):
    # Initialise arrays
    members = np.zeros((len(chains), 1))
    centres = np.zeros((len(chains), comp_dict[0].centroid.squeeze().shape[0]))
    variances = np.sum([obj.intraclust_var for obj in comp_dict.values()])
    # Loop over all the chains/superclusters
    for superclust, chain in enumerate(chains):
        # Calculate the centroid for the supercluster
        centres[superclust, :] = np.sum(
            np.multiply(base_centres[chain], base_members[chain]),
        axis=0)
        # Get the number of members in each supercluster
        members[superclust] = np.sum(base_members[chain])
    # We have taken the sum, so need to divide
    centres = np.divide(centres, members)
    # Calculate the intracluster variance for the superclusters
    wcss = np.sum([comp_dict[index].num_members *
                   np.dot(comp_dict[index].centroid.squeeze() -
                          centres[value], comp_dict[index].centroid.squeeze() - centres[value])
                   for index, value in enumerate(superclusts)])
    # print(wcss, variances)
    return wcss + variances


# @profile
def eval_mock(genotype, comp_dict, reduced_clust_nums, cnn_array, max_cnn, num_examples, data_dict, cnn_pairs,
              base_members, base_centres):
    # Identifies which components are connected (delta-evaluation)
    chains, superclusts = cluster_chains(genotype, data_dict, comp_dict, reduced_clust_nums)
    # Calculate the objectives
    CNN = objCNN(chains, superclusts, cnn_pairs, cnn_array, max_cnn)
    VAR = objVAR(chains, comp_dict, base_members, base_centres, superclusts)
    # Edge cases with large datasets, as single cluster solutions get given a 0 score
    if CNN < 0.00001: # Value that Garza/Handl uses, presumably to smooth small values
        CNN = 0
    # Divide by number of examples as per the paper and Mario's code
    return np.sum(VAR)/num_examples, CNN
