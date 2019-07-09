import itertools
import random

import igraph
import numpy as np


class PartialClust(object):
    # ID value iterates every time PartialClust is called
    # So we can ensure every object (base cluster) has a unique ID value
    id_value = itertools.count()
    
    cnn_array = None
    max_cnn = None
    max_var = None
    comp_dict = None
    cnn_pairs = None
    # Useful access to the info
    base_members = None
    base_centres = None

    def __init__(self, cluster):
        self.id = next(PartialClust.id_value) # Starts at cluster 0
        self.members = cluster
        self.num_members = len(cluster) # To avoid repeated len() calls for delta-VAR
        # Can call the two below if we convert them to method
        self.centroid = None # Call the function here?
        self.intraclust_var = None # Same as above

    @staticmethod
    def cnn_precomp(base_clusters, data_dict, argsortdists, L):
        cnn_array = np.zeros((len(base_clusters), len(base_clusters)))
        # pairs = np.zeros((int(len(base_clusters)*(len(base_clusters)/2)),),dtype=(int,2))
        # Initialise variables
        cnn_pair_list = []
        max_cnn = 0
        # Easier (one less nested loop, though same number of items)
        # than looping through base clusters - though it's equivalent
        for point in data_dict.values():
            # Get the L nearest neighbours
            # this matches the C++ code, where they consider the nearest neighbour to be the point itself
            # I've ignored this point, and just looked at the remaining L-1 points
            l_nns = argsortdists[point.id][1:L]
            # Get the base cluster number of the current point
            curr_number = point.base_cluster_num
            # Get the base cluster numbers of the L nearest neighbours
            nn_clust_nums = [data_dict[i].base_cluster_num for i in l_nns]
            # Loop over these and check if they're in different clusters
            for index, clust_num in enumerate(nn_clust_nums):
                if curr_number != clust_num:
                    # Mario's code uses <0.1, don't know why, need to ask
                    # That only really works if you're def using L=10
                    # We'll just use 0 now to make sure
                    if cnn_array[curr_number, clust_num] == 0:
                        cnn_pair_list.append((curr_number,clust_num))
                    # As we've skipped the first datapoint (itself)
                    # and to account for 0-based indexing and 1-based ranking
                    # we need to add 2 to the denominator to get the same as the C++
                    penalty = 1.0/(index+2.0)
                    # add the penalty
                    max_cnn += penalty
                    # Add contribution to relevant place in array
                    cnn_array[curr_number,clust_num] += penalty
                    # Make array symmetrical (as in Mario's code)
                    # Helps with the if statement above
                    cnn_array[clust_num,curr_number] = cnn_array[curr_number,clust_num]
        # print(np.sum(cnn_array), max_cnn)
        # print("Max conn:",max_cnn)
        return cnn_array, max_cnn, cnn_pair_list

    # @profile
    def var_precomp(self, data):
        # Makes the else irrelevant, but may be useful for memory issues if we tackle that later
        dist_meth = 'scipy'
        if dist_meth == 'scipy':
            # Scipy cdist, more precise, speed sometimes worse
            from scipy.spatial.distance import cdist
            centroid = np.mean(data[self.members],axis=0)[np.newaxis,:]
            dists = cdist(data[self.members],centroid,'sqeuclidean')
        else:
            # Sklearn metrics.pairwise.euclidean_distance
            # Fast, but not as precise (not guaranteed symmetry)
            from sklearn.metrics.pairwise import euclidean_distances
            centroid = np.mean(data[self.members],axis=0)[np.newaxis,:]
            dists = euclidean_distances(data[self.members],centroid,squared = True)
        return centroid, np.sum(dists) #np.einsum('ij->',dists)
    
    @classmethod
    def partial_clusts(cls, data, data_dict, argsortdists, L):
        cls.comp_dict = {}
        # Loop over our components as defined by delta
        for cluster in MOCKGenotype.base_clusters:
            # Create instance
            curr_cluster = cls(cluster)
            # Calculate centroid and VAR of component
            curr_cluster.centroid, curr_cluster.intraclust_var = cls.var_precomp(curr_cluster, data)
            # Assign id
            cls.comp_dict[curr_cluster.id] = curr_cluster
            # Label each point in this component in the data_dict
            for point in cluster:
                data_dict[point].base_cluster_num = curr_cluster.id
        # Calculate precomp for CNN objective
        cls.cnn_array, cls.max_cnn, cls.cnn_pairs = cls.cnn_precomp(
            MOCKGenotype.base_clusters, data_dict, argsortdists, L
        )
        # Used for speedy VAR calculation
        cls.base_members = np.asarray(
            [obj.num_members for obj in cls.comp_dict.values()]
            )[:, None]
        cls.base_centres = np.asarray(
            [obj.centroid for obj in cls.comp_dict.values()]
            ).squeeze()
        # Reset the counter
        cls.id_value = itertools.count()


class Datapoint(object):
    # We only ever handle one instance at a time
    # If we were to handle more, we'd need to actually move these into the __init__

    num_examples = None
    num_features = None
    labels = False
    data_name = None
    label_vals = None
    k_user = None

    def __init__(self, id_value, values):
        self.id = id_value
        self.values = values # what is this used for? Doesn't seem to be anything
        self.base_cluster_num = None

    @staticmethod
    def create_dataset(data_path, labels):
        # Auto identify delimiters and create array from data
        import csv
        with open(data_path) as file:
            dialect = csv.Sniffer().sniff(file.read())
            file.seek(0)
            # print(dialect.delimiter)
            data = np.genfromtxt(data_path, delimiter=dialect.delimiter, skip_header=0)
        
        # Assign name to dataset
        # Assuming file path split by / and filename only has one . character
        Datapoint.data_name = data_path.split("/")[-1].split(".")[0]
        print("Data Name:",Datapoint.data_name)

        # If we have labels, split array
        if labels:
            Datapoint.labels = True
            # Assuming labels are the final column
            label_vals = data[:, -1]
            Datapoint.label_vals = label_vals
            # We've stored the labels elsewhere so let's delete them from the data
            # Also avoids distance matrix issues!
            data = np.delete(data, -1, 1)
        # Create dictionary to store data points
        data_dict = {}
        # Loop over the dataset
        for id_value, row in enumerate(data):
            # Create current datapoint object
            curr_datapoint = Datapoint(id_value, row)
            # Assign label if present
            # if labels:
            # 	curr_datapoint.true_label = int(label_vals[id_value])
            # Store object in dictionary
            data_dict[curr_datapoint.id] = curr_datapoint

        [Datapoint.num_examples, Datapoint.num_features] = data.shape
        return data, data_dict

    @staticmethod
    def create_dataset_garza(data):
        if Datapoint.labels == True:
            Datapoint.label_vals = data[:, -1]
            data = np.delete(data, -1, 1)
        # Create dictionary to store data points
        data_dict = {}
        # Loop over the dataset
        for id_value, row in enumerate(data):
            # Create current datapoint object
            curr_datapoint = Datapoint(id_value, row)
            # Store object in dictionary
            data_dict[curr_datapoint.id] = curr_datapoint
        return data, data_dict


class MOCKGenotype:
    mst_genotype = None  # the MST genotype
    n_links = None
    degree_int = None  # Degree of interestingness of the MST
    interest_indices = None  # Indices of the most to least interesting links in the MST (formerly int_links_indices)
    interest_sorted_mst_genotype = None
    # Delta value
    #### In the future, can set this as the start
    #### And we redefine individual deltas as attributes if we have varying levels
    min_delta_val = None
    n_min_delta = None
    # Length of the reduced genotype
    reduced_length = None
    # Use these indices to slice from the genotype
    # Bulk update is easier with arrays, but could cause issues with DEAP
    # and forces us to rewrite nearly all of the other code...
    # Essentially equal to old int_links_indices[:relev_links_len]
    reduced_genotype_indices = None
    # For easier function evaluation
    reduced_cluster_nums = None
    # Base components
    base_genotype = None
    base_clusters = None

    def __init__(self):
        # Set full genotype as None - don't store a potentially long list unless we need to
        # (we always have the base as a class variable and can reconstruct)
        self.full_genotype = None        
        # The (reduced) genotype
        self.genotype = None
        # Number of clusters the individual defines
        self.num_clusts = None

    @classmethod
    def get_n_genes(cls, delta):
        return int(round((100 - delta) / 100 * cls.n_links, 0))

    @classmethod
    def get_random_delta(cls, min_delta, max_delta, precision=3):
        """Generates a random delta in a given interval"""
        delta_diff = max_delta - min_delta
        delta = round(random.random() * delta_diff + min_delta, precision)

        return delta

    @classmethod
    def delta_individual(cls, icls, min_delta, max_delta, delta=None, precision=3):
        """Generates an individual with a random delta in a given interval or value"""
        if delta is None:
            # Get the value for delta and the number of genes it represents
            delta = cls.get_random_delta(min_delta, max_delta, precision)

        n_genes = cls.get_n_genes(delta)

        ind = icls(cls.interest_sorted_mst_genotype[:n_genes])
        ind.delta = delta
        return ind

    @classmethod
    def rebuild_ind_genotype(cls, genotype):
        """
        :param genotype: list. Reduced individual's genotype
        :return: full individual's genotype
        """
        genotype_index = cls.interest_indices[:len(genotype)]
        full_genotype = cls.mst_genotype.copy()
        for i, idx in enumerate(genotype_index):
            full_genotype[idx] = genotype[i]
        return full_genotype

    @classmethod
    def expand_reduce_genotype(cls, parent, old_delta):
        # Number of genes that the new delta represents
        n = cls.get_n_genes(parent.delta)

        # Lock genes if delta has been increased
        if parent.delta > old_delta:
            parent[:] = parent[:n]

        # Otherwise, unlock new genes
        else:
            parent[:] += cls.interest_sorted_mst_genotype[len(parent):n]

        return parent

    @classmethod
    def setup_genotype_vars(cls):
        # Calculate the length of the reduced genotype
        cls.calc_red_length()
        # Find the indices of the most to least interesting links
        cls.interest_links_indices()
        # Store the indices that we need for our reduced genotype
        cls.reduced_genotype_indices = cls.interest_indices[:cls.reduced_length]
        # Identify the base components
        # i.e. set the most interesting links as specified by delta to be self-connecting so we create the base clusters
        cls.calc_base_genotype()
        # Identify these base clusters as the connected components of the defined base genotype
        cls.calc_base_clusters()
    
    @classmethod
    def interest_links_indices(cls):
        # Sort DI values to get the indices of most to least interesting links
        cls.interest_indices = np.argsort(
            -(np.asarray(MOCKGenotype.degree_int)), 
            kind='mergesort'
        ).tolist()
        cls.interest_sorted_mst_genotype = [cls.mst_genotype[i] for i in cls.interest_indices]

    @classmethod
    def calc_base_genotype(cls):
        """Create the base genotype, which is the encoded MST with the most interesting links removed (set to self-connecting)
        """
        cls.base_genotype = cls.mst_genotype[:]
        # Set the self-connecting links
        for index in cls.reduced_genotype_indices:
            cls.base_genotype[index] = index
    
    @classmethod
    def calc_base_clusters(cls):
        """Identify the base components
        """
        # Create a graph
        g = igraph.Graph()
        # Add the nodes
        g.add_vertices(len(MOCKGenotype.base_genotype))
        # Add the links
        g.add_edges(zip(
            range(len(MOCKGenotype.base_genotype)),
            MOCKGenotype.base_genotype)
        )
        # Set the components as a class attribute
        cls.base_clusters = list(g.components(mode="WEAK"))

    @classmethod
    def calc_reduced_clusts(cls, data_dict):
        """Identify the component IDs of where the link originates for the most interest points (i.e. the index). Speeds up objective evaluation.
        """
        cls.reduced_cluster_nums = [
            data_dict[i].base_cluster_num for i in MOCKGenotype.reduced_genotype_indices
        ]

    @staticmethod
    def replace_link(argsortdists, i, j, L):
        # Link can be replaced with L+1 options
        # L nearest neighbours and self-connecting link
        # Must exclude replacing with original link
        while True:
            # L+1 accounts for self-connecting link and L nearest neighbours
            new_j = random.choice(argsortdists[i][0:L+1])
            # Only break if we have made a new connection, otherwise try again
            if new_j != j:
                break
        return new_j
    
    @staticmethod
    def centroid_replace_link(argsortdists_cen, i, j, L_comp, data_dict):
        # Get the component ID of the data point
        point_comp = data_dict[i].base_cluster_num
        while True:
            # choose new component to mutate to
            new_comp = random.choice(argsortdists_cen[point_comp][0:L_comp+1])
            new_j = random.choice(PartialClust.comp_dict[new_comp].members)
            # Must be to a different link
            if new_j != j:
                break
        return new_j

    @staticmethod
    def neighbour_replace_link(component_nns, i, j, data_dict):
        # Get the component ID of the data point
        point_comp = data_dict[i].base_cluster_num
        while True:
            # Choose one of the nearest neighbours from a different component
            new_j = random.choice(component_nns[point_comp])
            if new_j != j:
                break
        return new_j

    @classmethod
    def calc_red_length(cls):
        """Calculate the reduced length of the genotype
        """
        cls.reduced_length = int(
            np.ceil(((100-MOCKGenotype.min_delta_val)/100) * Datapoint.num_examples)
        )
