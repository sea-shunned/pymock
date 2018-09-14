from sklearn import metrics
import numpy as np
import scipy.spatial as spt
from scipy.stats import rankdata

import igraph
# Add a try except for igraph?
# Shouldn't need this if the setup.py is done properly

def compDists(data1,data2):
    '''
    Compute distances between two datasets. Usually the same dataset will be passed as data1 and data2
    as we wish to get the dissimilarity matrix of the dataset

    :return: Distance array (or dissimilarity matrix) of the data
    '''
    return metrics.pairwise.euclidean_distances(data1, data2)
    
def compDists_sp(data):
    '''
    Compute the distance array of some data

    :return: Distance array (or dissimilarity matrix) of the data
    '''
    # Some say this should be faster, but I haven't really found to be
    # May be more memory efficient, so perhaps try this if that is a problem
    
    # Perhaps have a single function and try to catch a memory error, and use this distance if it fails
    return spt.distance.squareform(spt.distance.pdist(data,'euclidean'))

def createMST(distarray):
    # Create directed, weighted graph (loops=False means ignore diagonal)
    G = igraph.Graph.Weighted_Adjacency(distarray.tolist(), mode="DIRECTED", loops=False)

    # Get the MST
    # Does not randomise starting node(!)
    mst_ig = igraph.Graph.spanning_tree(G, weights=G.es["weight"], return_tree=False)

    # Create an array of infinities, so we get an error if we miss something!
    # We have one more vertex than edges, so +1
    gen_ig = np.full(len(mst_ig)+1, np.inf)

    # The general idea is to loop over every edge in the MST
        # and then fill in our genotype
        # if an edge has already been seen (not np.inf) then we fill in the reverse
    for i, edge in enumerate(mst_ig):
        edge_tup = G.es[edge].tuple
        
        if np.isinf(gen_ig[edge_tup[1]]):
            gen_ig[edge_tup[1]] = edge_tup[0]
        else:
            gen_ig[edge_tup[0]] = edge_tup[1]

    # As there is one more vertex than edges, find and fill in the missing one
    if np.isinf(gen_ig).any():
        index = np.where(np.isinf(gen_ig))[0][0]
        gen_ig[index] = np.where(gen_ig == index)[0][0]
        # gen_ig[index] = index
        
    # Cast types to integers
    gen_ig = gen_ig.astype(int)

    ## The below is for error checking
    H = igraph.Graph()
    H.add_vertices(len(gen_ig))
    H.add_edges([(index,edge) for index,edge in enumerate(gen_ig)])
    # Check we have a single connected component i.e. a fully-connected graph
    assert len(H.components(mode="WEAK")) == 1

    # Return as a list
    return gen_ig.tolist()

def normaliseDistArray(distarray):
    # Doing before hand saves one np.min() call
    max_val = np.max(distarray)
    min_val = np.min(distarray)
    denom = max_val - min_val

    distarray -= min_val
    distarray /= denom
    return distarray

def degreeInterest(mst_genotype, nn_rankings, distarray):
    # Calculate the degree of interest for each edge in the MST
    # This reads pretty much exactly as the formula (distances have been scaled)
    return [min(nn_rankings[i][j],nn_rankings[j][i])+distarray[i][j] for i,j in enumerate(mst_genotype)]

def interestLinksIndices(degree_int):
    '''
    Argsort the degree of interestingness list to get the indices of the most interesting links first

    Notes:
    Merge sort is stable and gives better ordering
    We use negative so that the lower indices appear first in the list

    :param degree_int: Degree of interestingness for each link in MST
    :return: Indices of most interesting links, in order of most to last (i.e. last is 0, as it connects to itself
             and will be the least interesting link)
    '''
    return np.argsort(-(np.asarray(degree_int)), kind='mergesort').tolist()

def LARfromMST(edgelist, mst):
    '''
    Convert the MST into a locus-based adjacency representation/encoding

    :param edgelist: Edgelist of the MST
    :param mst: The MST
    :return: A list in the format of a locus-based adjacency genotype
    '''
    # Initialise an array of the right length (number of nodes in the MST)
    indiv_array = np.zeros(len(list(mst.nodes())),).astype(int)

    # We first connect the first data item with itself
    # As the MST has one fewer edge than nodes, but LAR is of length #nodes
    indiv_array[0] = 0
    for edge in edgelist:
        indiv_array[edge[0]] = edge[1]
    return indiv_array.tolist()

def nnRankings(distarray, num_examples):
    """This function calculates the nearest neighbour ranking between all examples
    
    Arguments:
        distarray {np.array} -- Distance array of data
        num_examples {int} -- Number of data points
    
    Returns:
        [np.array] -- Nearest neighbour rankings for every data point
    """

    nn_rankings = np.zeros((num_examples,num_examples),dtype=int)
    for i, row in enumerate(distarray):
        nn_rankings[i] = rankdata(row, method='ordinal')-1 # minus 1 so that 0 rank is itself
    return nn_rankings

def nn_comps(num_examples, argsortdists, data_dict, L_comp):
    component_nns = np.zeros((num_examples, L_comp+1), dtype=int)

    for i, row_vals in enumerate(argsortdists):
        # Get the component ID of the row that we're in, so we know what to ignore
        start_comp_id = data_dict[i].base_cluster_num
        # Track what components have been seen
        comps_seen = set()
        comps_seen.add(start_comp_id)
        # Start with the point itself for self-connecting link
        nearest_l_ids = [i]
        # Loop over values in the row
        for val in row_vals:
            # Make local ref to the comp ID of current value
            curr_comp = data_dict[val].base_cluster_num
            # Check if component is both different and one we haven't seen
            if curr_comp != start_comp_id and curr_comp not in comps_seen:
                comps_seen.add(curr_comp)
                nearest_l_ids.append(val)
            # Break the loop when we have enough values
            if len(nearest_l_ids) == L_comp+1:
                break
        # Add values to the array
        component_nns[i,:] = nearest_l_ids
    return component_nns
