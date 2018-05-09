import itertools
import numpy as np
# import objectives
# from collections import OrderedDict

class PartialClust(object):
	# ID value iterates every time PartialClust is called
	# So we can ensure every object (base cluster) has a unique ID value
	id_value = itertools.count()
	
	conn_array = None
	max_conn = None

	# base_members = np.asarray([obj.num_members for obj in part_clust.values()])[:,None]
	# base_centres = np.asarray([obj.centroid for obj in part_clust.values()]).squeeze()

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


	def partClustCNN(base_clusters, data_dict, argsortdists, L):
		conn_array = np.zeros((len(base_clusters),len(base_clusters)))
		# pairs = np.zeros((int(len(base_clusters)*(len(base_clusters)/2)),),dtype=(int,2))

		# cnn_pair = 0
		cnn_pair_list = []

		max_conn = 0

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
					if conn_array[curr_number, clust_num] == 0:
						cnn_pair_list.append((curr_number,clust_num))

					# As we've skipped the first datapoint (itself)
					# and to account for 0-based indexing and 1-based ranking
					# we need to add 2 to the denominator to get the same as the C++
					penalty = 1.0/(index+2.0)
					
					max_conn += penalty

					# Add contribution to relevant place in array
					conn_array[curr_number,clust_num] += penalty

					# Make array symmetrical (as in Mario's code)
					# Helps with the if statement above
					conn_array[clust_num,curr_number] = conn_array[curr_number,clust_num]

		# print(np.sum(conn_array), max_conn)
		print("Max conn:",max_conn)
		return conn_array, max_conn, cnn_pair_list

	# @profile
	def partClustVAR(self, data):
		# Makes the else irrelevant, but may be useful for memory issues
		dist_meth = 'scipy'
		
		if dist_meth == 'scipy':
			# Scipy cdist, more precise, speed sometimes worse
			from scipy.spatial.distance import cdist
			centroid = np.mean(data[self.members],axis=0)[np.newaxis,:]

			dists = cdist(data[self.members],centroid,'sqeuclidean') #'euclidean'
			# u_v = data[self.members].squeeze() - centroid.squeeze()
			# dists2 = np.dot(u_v,np.transpose(u_v))

		else:
			# Sklearn metrics.pairwise.euclidean_distance
			# Fast, but not as precise (not guaranteed symmetry)
			from sklearn.metrics.pairwise import euclidean_distances
			centroid = np.mean(data[self.members],axis=0)[np.newaxis,:]
			dists = euclidean_distances(data[self.members],centroid,squared = True)
		return centroid, np.sum(dists) #np.einsum('ij->',dists)

# Not sure if this can even be put as a method (static method?) and if it even should be
# @profile
def partialClustering(base_clusters, data, data_dict, argsortdists, L):
	# partial_clustering = OrderedDict() # Don't think it's needed
	part_clust = {}

	for cluster in base_clusters:
		# Create cluster object
		curr_cluster = PartialClust(cluster)
		
		# Calculate centroid and intracluster variance for cluster object
		curr_cluster.centroid, curr_cluster.intraclust_var = PartialClust.partClustVAR(curr_cluster, data)
		
		# Add objective to dictionary, where key is the cluster ID number
		part_clust[curr_cluster.id] = curr_cluster

		# Assign base cluster number to relevant datapoint object
		for point in cluster:
			data_dict[point].base_cluster_num = curr_cluster.id

	# Once all objectives have been created, calculate max_conn score for partial clustering
	PartialClust.conn_array, PartialClust.max_conn, cnn_pairs = PartialClust.partClustCNN(base_clusters, data_dict, argsortdists, L)

	# for datapoint in data_dict.values():
	# 	obj = part_clust[datapoint.base_cluster_num]
	# # 	print(datapoint.id, obj.id, obj.intraclust_var, obj.centroid[:,:3], obj.num_members)
	# 	print(datapoint.id, obj.id)

	PartialClust.base_members = np.asarray([obj.num_members for obj in part_clust.values()])[:,None]
	PartialClust.base_centres = np.asarray([obj.centroid for obj in part_clust.values()]).squeeze()

	print("No. base clusters:",len(part_clust))
	print("Length cnn pair list:",len(cnn_pairs))

	return part_clust, cnn_pairs

class Dataset(object):
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
		self.values = values
		self.base_cluster_num = None	
		# self.true_label = None # See discussion above

def createDataset(data_path, labels):
	# Auto identify delimiters and create array from data
	import csv
	with open(data_path) as file:
		dialect = csv.Sniffer().sniff(file.read())
		file.seek(0)
		# print(dialect.delimiter)
		data = np.genfromtxt(data_path, delimiter = dialect.delimiter, skip_header=0)
	
	# Assign name to dataset
	# Assuming file path split by / and filename only has one . character
	Dataset.data_name = data_path.split("/")[-1].split(".")[0]
	print("Data Name:",Dataset.data_name)

	# If we have labels, split array
	if labels:
		Dataset.labels = True

		# Assuming labels are the final column
		label_vals = data[:,-1]
		Dataset.label_vals = label_vals

		# We've stored the labels elsewhere so let's delete them from the data
		# Also avoids distance matrix issues!
		data = np.delete(data, -1, 1)

	# Create dictionary to store data points
	data_dict = {}

	# Loop over the dataset
	for id_value, row in enumerate(data):

		# Create current datapoint object
		curr_datapoint = Dataset(id_value, row)

		# Assign label if present
		# if labels:
		# 	curr_datapoint.true_label = int(label_vals[id_value])

		# Store object in dictionary
		data_dict[curr_datapoint.id] = curr_datapoint

	[Dataset.num_examples, Dataset.num_features] = data.shape
	return data, data_dict

def createDatasetGarza(data):
	if Dataset.labels == True:
		Dataset.label_vals = data[:,-1]

		data = np.delete(data, -1, 1)

	# Create dictionary to store data points
	data_dict = {}

	# Loop over the dataset
	for id_value, row in enumerate(data):

		# Create current datapoint object
		curr_datapoint = Dataset(id_value, row)

		# Store object in dictionary
		data_dict[curr_datapoint.id] = curr_datapoint

	return data, data_dict

# Possibly useful class to implement some way down the line
# Currently doesn't feel worth the effort as everything works reasonably well
class Genotype(list):
	mst_genotype = None # the MST genotype
	unfixed_indices = None # To replace int_links_indices

	def __init__(self, index, value):
		self.index = index # Gene value in genotype ## Is this necessary?
		self.value = value # Either have as tuple or just use the value (and then rename)
		self.fixed = False # Identify if the gene is fixed or not
		self.newly_unfixed = False # For adaptive delta

	@staticmethod
	def calcDI():
		pass

	@staticmethod
	def baseGenotype():
		pass

	def replaceLink():
		pass
		# We could actually use self here right? As we are replacing a gene

	## This needs some redesign
	# What do we actually want to use this for? If it is for keeping track of fixed/unfixed links, and newly unfixed for fair mutation
	# Then the class is more the MST
	# Consider it like a dictionary with nested keys
	# The first key each time is the root node for the link, or the index of the gene in the genotype
	# Then we have some static, unique values here:
		# The MST
		# The base genotype, where some links in the MST change to self-connecting links
	# We do not need to store 
	# It may be useful for our solutions to have additional attributes, like C++ MOCK does
		# Such as the number of clusters in the solution
		# But this is different to the MST thing above, in a way
		# We could have one class for this, where instances are our actual solutions that have these attributes (we create them in initial pop)
			# Shouldn't be a problem with DEAP
		# And everything else (mentioned above) is static


	# Could we actually implement some of the functionality we want from this class into the existing dataset class?
	# Each datapoint is a node on the graph, after all, so we can just give it a value (what it points to in the MST)
	# And then we can track whether it is fixed or not - or more importantly, if it needs fair mutation