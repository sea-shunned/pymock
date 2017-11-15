import precompute
import initialisation
import objectives
import operators
import classes
import numpy as np
from itertools import count

from os import cpu_count

from deap import base
from deap import creator
from deap import tools
from deap.benchmarks.tools import hypervolume

import multiprocessing

# Run outside of multiprocessing scope
creator.create("Fitness", base.Fitness, weights=(-1.0, -1.0)) #(VAR, CNN)
creator.create("Individual", list, fitness=creator.Fitness)

# @profile # for line_profiler
def main(data_path, delta_val):
	# print("Delta:",delta_val)

	# If labels are present (will move to a argparse thing eventually)
	labels = True

	# Load data
	data, data_dict = classes.createDataset(data_path, labels)

	######## Parameters ########
	# Neighbourhood size
	L = 10

	# Population size
	num_indivs = 100

	# Reduced genotype length
	relev_links_len = initialisation.relevantLinks(delta_val, classes.Dataset.num_examples)
	# fixed_links_len = num_examples - relev_links_len

	#### relev_links_len needs a rename to more accurately describe that it is the reduced genotype length

	# k_user is the double the expected number of clusters in the dataset
	k_user = 10 # will move to a argparse thing eventually

	############## TO DO ##############
	######## Precomputation ########
	### Should I consider putting all of the below in it's own script/function?
	### A precomputation function or something that does all of this
	### Or at least abstract out this vars_check business into precomputation

	import time # Just to see how long sections take

	distarray = precompute.compDists(data,data)
	distarray = precompute.normaliseDistArray(distarray)
	argsortdists = np.argsort(distarray, kind='mergesort')
	nn_rankings = precompute.nnRankings(distarray, classes.Dataset.num_examples)
	mst_genotype = precompute.createMST(distarray)

	##### Change DI calc in this func #####
	# Calculate the degree of interestingness of each link
	degree_int, I, int_bool = precompute.interestLinks(mst_genotype, L, argsortdists)

	# Return an array with the indices of the most interesting links first
	int_links_indices = precompute.interestLinksIndices(degree_int)

	base_genotype, base_clusters = initialisation.baseGenotype(mst_genotype, int_links_indices, relev_links_len)
	part_clust, cnn_pairs = classes.partialClustering(base_clusters, data, data_dict, argsortdists, L)
	conn_array, max_conn = classes.PartialClust.conn_array, classes.PartialClust.max_conn
	reduced_clust_nums = [data_dict[i].base_cluster_num for i in int_links_indices[:relev_links_len]]

	print("\n")	
	print("Delta:", delta_val)
	print("Max Conn:", max_conn)
	print("No. base clusters:",len(base_clusters), len(part_clust))
	print("Reduced clust nums:", len(reduced_clust_nums))
	# print("\n")

	toolbox = base.Toolbox()
	toolbox.register("evaluate", objectives.evalMOCK, part_clust = part_clust, reduced_clust_nums = reduced_clust_nums, conn_array = conn_array, max_conn = max_conn, num_examples = classes.Dataset.num_examples, data_dict=data_dict, cnn_pairs=cnn_pairs)

	indiv = mst_genotype[:]
	mst_red_indiv = [indiv[i] for i in int_links_indices[:relev_links_len]]

	print("MST Indiv Fitness:",toolbox.evaluate(mst_red_indiv))
	# print("Random Indiv Fitness:",toolbox.evaluate(mst_red_indiv))

	classes.PartialClust.id_value = count()

	delta_val = 80
	relev_links_len = initialisation.relevantLinks(delta_val, classes.Dataset.num_examples)
	base_genotype, base_clusters = initialisation.baseGenotype(mst_genotype, int_links_indices, relev_links_len)
	part_clust, cnn_pairs = classes.partialClustering(base_clusters, data, data_dict, argsortdists, L)
	conn_array, max_conn = classes.PartialClust.conn_array, classes.PartialClust.max_conn
	reduced_clust_nums = [data_dict[i].base_cluster_num for i in int_links_indices[:relev_links_len]]

	print("Delta:", delta_val)
	print("Max Conn:", max_conn)
	print("No. base clusters:",len(base_clusters), len(part_clust))
	print("Reduced clust nums:", len(reduced_clust_nums))
	# print("\n")

	toolbox.unregister("evaluate")
	toolbox.register("evaluate", objectives.evalMOCK, part_clust = part_clust, reduced_clust_nums = reduced_clust_nums, conn_array = conn_array, max_conn = max_conn, num_examples = classes.Dataset.num_examples, data_dict=data_dict, cnn_pairs=cnn_pairs)
	mst_red_indiv = [indiv[i] for i in int_links_indices[:relev_links_len]]
	print("MST Indiv Fitness:",toolbox.evaluate(mst_red_indiv))
	# print("Random Indiv Fitness:",toolbox.evaluate(mst_red_indiv))

	classes.PartialClust.id_value = count()

	print("\n")

if __name__ == '__main__':
	import os
	import pandas as pd

	delta_vals = [50,70,90]

	base_path = os.getcwd()

	## Set data_folder
	data_folder = base_path+"/data/data_handl/"
	results_folder = base_path+"/results/"

	data_name = "10d-20c-no0.dat"
	data_path = data_folder+data_name

	for run, delta in enumerate(delta_vals):
		main(data_path, delta)



	# The issue is that all indivs in the population are stored in their reduced form
	# When changing delta, every individual now needs to change length
	# Need to see if we should actually store the whole individual in the population
	# And then just take what we need from each (i.e. reduce it) when mutating, evaluating etc.