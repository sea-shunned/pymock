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
import main_base # for DEAP indiv declaration
import artif_carryon
import artif_hypermutspec
import artif_hypermutall
import artif_reinit
import artif_fairmut

import adaptive_funcs

# Get our current wd as the base path
basepath = os.getcwd()

# Set paths for datasets
data_folder = os.path.join(basepath, "data")+os.sep
synth_data_folder = os.path.join(data_folder, "synthetic_datasets")+os.sep
real_data_folder = os.path.join(data_folder, "UKC_datasets")+os.sep

synth_data_files = sorted(glob.glob(synth_data_folder+'*.data'))
real_data_files = sorted(glob.glob(real_data_folder+'*.txt'))

results_folder = os.path.join(basepath,"results","artif")+os.sep

# data_files = [synth_data_files[1]] + [synth_data_files[93]] + [synth_data_files[167]] + [synth_data_files[214]] + [synth_data_files[228]] + [synth_data_files[315]] + [synth_data_files[306]] + real_data_files[1:2] + real_data_files[-1:]

# data_files = sorted(glob.glob(synth_data_folder+'*_9_*.data'))
data_files = sorted(glob.glob(real_data_folder+'*.txt'))
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
sr_vals = [5,1]

# Parameters across all strategies
L = 10
num_indivs = 100
num_gens = 100
delta_reduce = 1

funcs = [main_base.main, artif_carryon.main, artif_hypermutspec.main, artif_hypermutall.main, artif_reinit.main, artif_fairmut.main]

save_results = True

fitness_cols = ["VAR", "CNN", "Run"]

# Dynamic interval trigger_gen list
# trigger_gen_list = [adaptive_funcs.triggerGens_interval(num_gens) for i in range(num_runs)]

# Static interval trigger_gen list
# trigger_gen_list = [[12, 40, 63, 87], [12, 40, 64, 80], [27, 41, 60, 80], [13, 38, 60, 75], [19, 31, 64, 75], [11, 41, 51, 71], [21, 39, 64, 74], [10, 32, 50, 75], [23, 31, 50, 73], [29, 39, 66, 70], [23, 31, 52, 78], [20, 44, 50, 82], [11, 34, 55, 78], [22, 42, 64, 72], [11, 39, 60, 72], [28, 49, 66, 74], [20, 31, 67, 89], [27, 36, 66, 73], [14, 30, 57, 78], [12, 42, 65, 79], [18, 33, 50, 88], [14, 36, 51, 83], [20, 42, 61, 74], [25, 48, 56, 70], [10, 43, 64, 83], [20, 36, 60, 74], [26, 44, 61, 81], [10, 44, 66, 88], [29, 45, 57, 76], [14, 31, 54, 87]]

# # Static random trigger_gen list
trigger_gen_list = [[33, 43, 53, 84], [13, 45, 62, 72], [32, 54, 76, 88], [9, 71, 81, 91], [18, 28, 39, 49], [42, 52, 62, 72], [14, 24, 59, 77], [21, 40, 50, 60], [14, 27, 74, 89], [49, 60, 70, 80], [17, 27, 47, 73], [52, 62, 72, 82], [11, 21, 31, 87], [11, 21, 82, 92], [19, 29, 55, 66], [32, 42, 67, 81], [15, 78, 89, 99], [11, 36, 46, 56], [35, 60, 70, 80], [54, 74, 84, 94], [24, 34, 72, 82], [42, 52, 64, 74], [15, 30, 57, 67], [14, 34, 44, 84], [13, 27, 44, 73], [25, 35, 45, 55], [43, 53, 63, 73], [29, 39, 49, 59], [49, 59, 69, 79], [24, 51, 61, 81]]


assert len(trigger_gen_list) >= num_runs, "Too many runs for number of available seeds"
print("Trigger gen list:", trigger_gen_list)

print('Number of delta values to test:', len(sr_vals))
print("Number of runs per delta value:", num_runs)
print("Number of datasets:", len(data_files))
print("Number of strategies:", len(funcs))
print("Number of total MOCK Runs:", len(data_files)*len(sr_vals)*num_runs*len(funcs),"\n")

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

	results_folder_data = results_folder+classes.Dataset.data_name+os.sep

	# Add square root delta values
	delta_vals = [100-((100*i*np.sqrt(classes.Dataset.num_examples))/classes.Dataset.num_examples) for i in sr_vals]

	# Print some outputs about the experiment configuration
	print("Delta values to test:", delta_vals, "("+str(len(delta_vals))+")")

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

	# Generate SR5 HV ref point to avoid issues
	# _,_, HV_ref ,_,_,_ = main_base.main(data, data_dict, 100-((100*5*np.sqrt(classes.Dataset.num_examples))/classes.Dataset.num_examples), HV_ref, argsortdists, nn_rankings, mst_genotype, int_links_indices, L, num_indivs, num_gens, delta_reduce)

	for index_d, delta in enumerate(delta_vals):
		print("\nTesting delta =",delta, "(sr"+str(sr_vals[index_d])+")")

		# Create tuple of arguments
		args = data, data_dict, delta, HV_ref, argsortdists, nn_rankings, mst_genotype, int_links_indices, L, num_indivs, num_gens, delta_reduce

		if HV_ref == None:
			first_run = True

		else:
			first_run = False

		for func in funcs:
			strat_name = func.__globals__["__file__"].split("/")[-1].split(".")[0].split("_")[-1]
			print(strat_name)
			print(func.__globals__["__file__"].split("/")[-1].split(".")[0].split("_")[-1])
			
			# Don't do sr5 for any of the artif scripts
			if strat_name != "base" and sr_vals[index_d]==5:
				# print("\n",strat_name, sr_vals[index_d], delta,"\n")
				continue

			# # Don't do sr1 for base MOCK
			# if strat_name == "main_base" and sr_vals[index_d]==1:
			# 	continue

			# Create arrays to save results for the given function
			fitness_array = np.empty((num_indivs*num_runs, len(fitness_cols)))
			hv_array = np.empty((num_gens, num_runs))
			ari_array = np.empty((num_indivs, num_runs))
			numclusts_array = np.empty((num_indivs, num_runs))
			time_array = np.empty(num_runs)
			delta_triggers = []

			print("\nStrategy:",strat_name, "Delta: sr",sr_vals[index_d])

			for run in range(num_runs):
				random.seed(seeds[run])
				print("\nSeed number:",seeds[run])
				print("HV ref:", HV_ref)

				print("Run",run,"with", strat_name)
				if strat_name == "base":
					start_time = time.time()
					pop, HV, HV_ref_temp, int_links_indices_spec, relev_links_len, adapt_gens = func(*args)
					end_time = time.time()
				else:	
					start_time = time.time()
					pop, HV, HV_ref_temp, int_links_indices_spec, relev_links_len, adapt_gens = func(*args, trigger_gens=trigger_gen_list[run])
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
				np.savetxt(filename+"-fitness-sr"+str(sr_vals[index_d])+"-random.csv", fitness_array, delimiter=",")
				np.savetxt(filename+"-hv-sr"+str(sr_vals[index_d])+"-random.csv", hv_array, delimiter=",")
				np.savetxt(filename+"-ari-sr"+str(sr_vals[index_d])+"-random.csv", ari_array, delimiter=",")
				np.savetxt(filename+"-numclusts-sr"+str(sr_vals[index_d])+"-random.csv", numclusts_array, delimiter=",")
				np.savetxt(filename+"-time-sr"+str(sr_vals[index_d])+"-random.csv", time_array, delimiter=",")

				# Pickle delta triggers
				# No triggers for normal delta-MOCK
				if strat_name != "base":
					with open(filename+"-triggers-sr"+str(sr_vals[index_d])+"-random.csv","w") as f:
					# 	pickle.dump(delta_triggers, f)
						writer=csv.writer(f)
						writer.writerows(delta_triggers)

		# Modify the below for specific dataset folder
		# np.savetxt(results_path+classes.Dataset.data_name[:-15]+"_eaf_"+str(delta)+".csv", arr, delimiter=" ")

	print(classes.Dataset.data_name + " complete! Took",time.time()-file_time,"seconds \n")


### Graph/Analysis after all the data is done
# Useful as we still have the experiment info available in memory
# e.g. num runs etc.