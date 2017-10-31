import os
import pandas as pd
import csv
# import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import main_base
import classes
import precompute
import time

import random

#### Might need to put the DEAP stuff here as if we import multiple scripts we define an indiv multple times


# Set paths for datasets
synth_data_folder = "/home/cshand/Documents/Delta-MOCK-master/Datasets/synthetic_datasets/"
real_data_folder = "/home/cshand/Documents/Delta-MOCK-master/Datasets/UKC_datasets/"

synth_data_files = glob.glob(synth_data_folder+'tevc_20_*.data')
real_data_files = glob.glob(real_data_folder+'*.txt')

results_basepath = "/home/cshand/Documents/PhD_MOCK/adaptive_delta/results/"

# print(len(synth_data_files),len(real_data_files), type(synth_data_files))

# data_files = synth_data_files + real_data_files
# synth_data_files.clear()
# real_data_files.clear()

# print(len(data_files))

# data_files = synth_data_files[:3] + real_data_files[:1]
data_files = synth_data_files[:2]
print(data_files)

num_runs = 4
seeds = [random.randint(0,10000)*i for i in range(num_runs)]
# Pickle (save) the seeds here

# Set range of delta values to test for each file
# delta_vals = [i for i in range(90,100,5) for _ in range(num_runs)]
delta_vals = [i for i in range(0,100,90)]

print("Delta values to test:", delta_vals)
print("Number of runs per delta value:", num_runs)
print("Number of datasets:", len(data_files))
print("Number of total MOCK Runs:", len(data_files)*len(delta_vals)*num_runs)

## Below will be kept in the actual main function
## But just put here for a checklist
# Check that 100 indivs and 100 generations is OK
L = 10
num_indivs = 100
# Check crossover probability
# Or do I want to put this stuff here? Eh.

df_cols = ["VAR", "CNN", "Run"]

for file_path in data_files:
	import classes # May need to put this here to ensure counts etc. are reset - TEST THIS
	classes.Dataset.data_name = file_path.split("/")[-1].split(".")[0]
	print("Testing:",classes.Dataset.data_name)

	results_path = results_basepath+classes.Dataset.data_name[:-15]+"/"

	try:
		os.mkdir(results_path)

	except FileExistsError:
		pass

	# USE A TRY EXCEPT HERE TO CREATE A DATA FOLDER
	# '_'.join()
	# ['_'.join(file.split("/")[-1].split("_")[:-4]) for file in files]

	# Get header info (only for Mario's data!)
	with open(file_path) as file:
		head = [int(next(file)[:-1]) for _ in range(4)]

	# Read the data into an array
	data = np.genfromtxt(file_path, delimiter="\t", skip_header=4)
	# print(data.shape)

	# Set the values for the data
	classes.Dataset.num_examples = head[0] # Num examples
	classes.Dataset.num_features = head[1] # Num features/dimensions
	classes.Dataset.k_user = head[3] # Num real clusters

	# Do we have labels?
	if head[2] == 1:
		classes.Dataset.labels = True
	else:
		classes.Dataset.labels = False
	# If labels=True, then we can calc ARI, otherwise we can't, correct this!

	# Remove labels if present and create data_dict
	data, data_dict = classes.createDatasetGarza(data)

	# Precomputation for this dataset
	print("Starting precomputation...")

	# start_time = time.time()
	distarray = precompute.compDists(data, data)
	# end_time = time.time()
	# print("sklearn took",end_time-start_time,"seconds")
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
	print("Precomputation done!")
	# Which of the above do I need to pass?
	# argsortdists, nn_rankings, mst_genotype, int_links_indices
	
	### To add here ###
	#HV_ref
	HV_ref = None

	# for run, delta in enumerate(delta_vals):
	for delta in delta_vals:

		# Initialise df
		# df = pd.DataFrame(columns=df_cols)

		arr = np.zeros((num_indivs*num_runs,len(df_cols)))

		for run in range(num_runs):
			random.seed(seeds[run])

			print("HV ref:", HV_ref)

			if HV_ref == None:
				first_run = True

			else:
				first_run = False

			start_time = time.time()
			pop, logbook, _, _, HV, ea_time, final_pop_metrics, HV_ref_temp = main_base.main(data, data_dict, delta, HV_ref, argsortdists, nn_rankings, mst_genotype, int_links_indices, L, num_indivs)
			end_time = time.time()
			print("Run "+str(run)+" for d="+str(delta)+" complete (Took",end_time-start_time,"seconds)")
			if first_run:
				HV_ref = HV_ref_temp

			# print(final_pop_metrics,"\n")

			# Check logbook to see how the form looks, and if we can manipulate it into a dataframe
			# May want to just loop through the pop and get the fitness values
			# And add the run number to the 3rd column

			# This is overwriting
			ind = num_indivs*run

			arr[ind:ind+num_indivs,0:3] = [indiv.fitness.values+(run,) for indiv in pop]

		# Modify the below for specific dataset folder
		np.savetxt(results_path+classes.Dataset.data_name[:-15]+"_eaf_"+str(delta)+".csv", arr, delimiter=" ")