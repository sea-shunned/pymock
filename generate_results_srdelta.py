import os
import pandas as pd
import csv
# import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import time
import random

# MOCK files needed for precomputation
import classes
import precompute
import evaluation

# Import strategies here
import main_base
import main_carryon
import main_hypermutspec
import main_hypermutall
import main_reinit
import main_fairmut

# Get our current wd as the base path
basepath = os.getcwd()

# Set paths for datasets
data_folder = basepath+"/data/"
synth_data_folder = data_folder+"synthetic_datasets/"
real_data_folder = data_folder+"UKC_datasets/"

# synth_data_files = glob.glob(synth_data_folder+'tevc_20_10_6_*.data')
# synth_data_files = glob.glob(synth_data_folder+'tevc_50_40_7_*.data')
# synth_data_files = glob.glob(synth_data_folder+'tevc_100_40_3_*.data')

synth_data_files = glob.glob(synth_data_folder+'*.data')
real_data_files = glob.glob(real_data_folder+'*.txt')

results_folder = basepath+"/results/"

data_files = synth_data_files[:3] + [synth_data_files[8]] + [synth_data_files[11]] + [synth_data_files[19]] + [synth_data_files[37]] + real_data_files[:1]

# synth_data_files = glob.glob(synth_data_folder+'tevc_20_10_6_*.data')
# data_files = synth_data_files

print(data_files)

# Specify the number of runs
num_runs = 30

# Randomly generated numbers to use as the fixed seeds
# 50 unique seeds, should be enough as unlikely to run more than 50 times
seeds = [11, 472, 6560, 15159, 25560, 4062, 24052, 56256, 66978, 64800, 6413, 119628, 2808, 115892, 118905, 140784, 47889, 26838, 142234, 139740, 163359, 127666, 10764, 62256, 191875, 30472, 66150, 169008, 285012, 4890, 187488, 223680, 18480, 42738, 210280, 173916, 111851, 289940, 159510, 250760, 31160, 143976, 70907, 142076, 311715, 68034, 49491, 144768, 376663, 354300]

# Ensure we have unique seed numbers
assert len(seeds) == len(set(seeds)), "Non-unique seed numbers"
# Ensure that we have the right number of seeds for the number of runs
assert len(seeds) >= num_runs, "Too many runs for number of available seeds"

# Set range of delta values to test for each file
# delta_vals = [i for i in range(90,99,3)]
# delta_vals = []

# Square root values for delta
# Reverse to ensure lowest delta is first (in case of issues with HV ref point)
sr_vals = [5,2,1]

# Parameters across all strategies
L = 10
num_indivs = 100
num_gens = 100
delta_reduce = 1

funcs = [main_base.main, main_carryon.main, main_hypermutspec.main, main_hypermutall.main, main_reinit.main, main_fairmut.main]
# funcs = [main_fairmut.main, main_base.main]
funcs = [main_fairmut.main]
save_results = True

fitness_cols = ["VAR", "CNN", "Run"]


for file_path in data_files:
	file_time = time.time()
	import classes # May need to put this here to ensure counts etc. are reset - TEST THIS
	classes.Dataset.data_name = file_path.split("/")[-1].split(".")[0][:-15]
	
	# Correction for real dataset
	if classes.Dataset.data_name == "":
		classes.Dataset.data_name = file_path.split("/")[-1].split(".")[0]

	print("Testing:",classes.Dataset.data_name)

	# USE A TRY EXCEPT HERE TO CREATE A DATA FOLDER
	# '_'.join()
	# ['_'.join(file.split("/")[-1].split("_")[:-4]) for file in files]

	# Get header info (only for Mario's data!)
	with open(file_path) as file:
		head = [int(next(file)[:-1]) for _ in range(4)]

	# Read the data into an array
	data = np.genfromtxt(file_path, delimiter="\t", skip_header=4)

	# Set the values for the data
	classes.Dataset.num_examples = head[0] # Num examples
	classes.Dataset.num_features = head[1] # Num features/dimensions
	classes.Dataset.k_user = head[3] # Num real clusters

	# Do we have labels?
	if head[2] == 1:
		classes.Dataset.labels = True
	else:
		classes.Dataset.labels = False

	# Remove labels if present and create data_dict
	data, data_dict = classes.createDatasetGarza(data)

	results_folder_data = results_folder+classes.Dataset.data_name+"/"

	# Add square root delta values
	delta_vals = [100-((100*i*np.sqrt(classes.Dataset.num_examples))/classes.Dataset.num_examples) for i in sr_vals]

	# Print some outputs about the experiment configuration
	print("Delta values to test:", delta_vals, "("+str(len(delta_vals))+")")
	print("Number of runs per delta value:", num_runs)
	print("Number of datasets:", len(data_files))
	print("Number of strategies:", len(funcs))
	print("Number of total MOCK Runs:", len(data_files)*len(delta_vals)*num_runs*len(funcs),"\n")


	###	Try to create a folder for results, group by the k & d
	if not os.path.isdir(results_folder_data):
		print("Created a results folder for dataset "+classes.Dataset.data_name)
		os.makedirs(results_folder_data)

	# Precomputation for this dataset
	print("Starting precomputation...")

	# start_time = time.time()
	distarray = precompute.compDists(data, data)
	# end_time = time.time()
	distarray = precompute.normaliseDistArray(distarray)
	print("Distance array done!")

	argsortdists = np.argsort(distarray, kind='mergesort')
	nn_rankings = precompute.nnRankings(distarray, classes.Dataset.num_examples)
	print("NN rankings done!")

	start_time = time.time()
	mst_genotype = precompute.createMST(distarray)
	end_time = time.time()
	print("MST done! (Took",end_time-start_time,"seconds)")

	degree_int = precompute.degreeInterest(mst_genotype, L, nn_rankings, distarray)
	int_links_indices = precompute.interestLinksIndices(degree_int)
	print("DI done!")
	print("Precomputation done!\n")

	HV_ref = None

	for index_d, delta in enumerate(delta_vals):
		print("\nTesting delta =",delta)

		# Create tuple of arguments
		args = data, data_dict, delta, HV_ref, argsortdists, nn_rankings, mst_genotype, int_links_indices, L, num_indivs, num_gens, delta_reduce

		if HV_ref == None:
			first_run = True

		else:
			first_run = False

		##### What numbers do I want and where???? #####

		for func in funcs:
			# Create arrays to save results for the given function
			fitness_array = np.empty((num_indivs*num_runs, len(fitness_cols)))
			hv_array = np.empty((num_gens, num_runs))
			ari_array = np.empty((num_indivs, num_runs))
			numclusts_array = np.empty((num_indivs, num_runs))
			time_array = np.empty(num_runs)
			delta_triggers = []

			strat_name = func.__globals__["__file__"].split("/")[-1].split(".")[0]

			for run in range(num_runs):
				random.seed(seeds[run])
				print("\nSeed number:",seeds[run])
				print("HV ref:", HV_ref)

				print("Run",run,"with", strat_name)
				start_time = time.time()
				pop, HV, HV_ref_temp, int_links_indices_spec, relev_links_len, adapt_gens = func(*args)
				end_time = time.time()
				print("Run "+str(run)+" for d="+str(delta)+" (sr"+str(sr_vals[index_d])+") complete (Took",end_time-start_time,"seconds)")

				if first_run:
					HV_ref = HV_ref_temp

				# Add fitness values
				ind = num_indivs*run
				fitness_array[ind:ind+num_indivs,0:3] = [indiv.fitness.values+(run+1,) for indiv in pop]

				# Calculate number of clusters and the ARI for each individual in the final pop
				numclusts, aris = evaluation.finalPopMetrics(pop, mst_genotype, int_links_indices_spec, relev_links_len)

				# Assign these values
				numclusts_array[:,run] = numclusts
				ari_array[:,run] = aris

				# Assign the HV
				hv_array[:,run] = HV

				# Assign the time taken
				time_array[run] = end_time - start_time

				delta_triggers.append(adapt_gens)

			###### Create a folder for graphs, and save the graphs we make into that
			###### Best to do that here and just have the graph func return a graph
				# Easier to aggregate here
				# Easier to save graphs for individual runs from within the main func, or aggregate over a single run with multiple funcs here

				# print(func.__globals__["__file__"].split("/")[-1].split(".")[0])

			# Save the arrays here
			# np.savetxt(fname,array,delimiter=',')
			filename = "-".join([results_folder_data+classes.Dataset.data_name,strat_name])

			if save_results:
				# Save array data
				np.savetxt(filename+"-fitness-sr"+str(sr_vals[index_d])+"dh3.csv", fitness_array, delimiter=",")
				np.savetxt(filename+"-hv-sr"+str(sr_vals[index_d])+"dh3.csv", hv_array, delimiter=",")
				np.savetxt(filename+"-ari-sr"+str(sr_vals[index_d])+"dh3.csv", ari_array, delimiter=",")
				np.savetxt(filename+"-numclusts-sr"+str(sr_vals[index_d])+"dh3.csv", numclusts_array, delimiter=",")
				np.savetxt(filename+"-time-sr"+str(sr_vals[index_d])+"dh3.csv", time_array, delimiter=",")

				# Pickle delta triggers
				# No triggers for normal delta-MOCK
				if strat_name != "main_base":
					with open(filename+"-triggers-sr"+str(sr_vals[index_d])+"dh3.csv","w") as f:
					# 	pickle.dump(delta_triggers, f)
						writer=csv.writer(f)
						writer.writerows(delta_triggers)

		# Modify the below for specific dataset folder
		# np.savetxt(results_path+classes.Dataset.data_name[:-15]+"_eaf_"+str(delta)+".csv", arr, delimiter=" ")

	print(classes.Dataset.data_name + " complete! Took",time.time()-file_time,"seconds \n")


### Graph/Analysis after all the data is done
# Useful as we still have the experiment info available in memory
# e.g. num runs etc.
