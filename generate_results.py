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

# Import strategies here
import main_base
import main_carryon


# Get our current wd as the base path
basepath = os.getcwd()

# Set paths for datasets
data_folder = basepath+"/data/"
synth_data_folder = data_folder+"synthetic_datasets/"
real_data_folder = data_folder+"UKC_datasets/"

synth_data_files = glob.glob(synth_data_folder+'tevc_20_10_6_*.data')
# synth_data_files = glob.glob(synth_data_folder+'tevc_50_40_7_*.data')
# synth_data_files = glob.glob(synth_data_folder+'tevc_100_40_3_*.data')
# real_data_files = glob.glob(real_data_folder+'*.txt')

results_folder = basepath+"/results/"

# data_files = synth_data_files[:3] + real_data_files[:1]
data_files = synth_data_files

num_runs = 2
# seeds = [random.randint(0,10000)*i for i in range(num_runs)]
seeds = [11, 1000]
# Pickle (save) the seeds here

# Set range of delta values to test for each file
delta_vals = [i for i in range(90,97,5)]

print("Delta values to test:", delta_vals)
print("Number of runs per delta value:", num_runs)
print("Number of datasets:", len(data_files))
print("Number of total MOCK Runs:", len(data_files)*len(delta_vals)*num_runs)

## Below will be kept in the actual main function
## But just put here for a checklist
# Check that 100 indivs and 100 generations is OK
############ Should try to add num_indivs and num_gens here
L = 10
num_indivs = 100
# Check crossover probability
# Or do I want to put this stuff here? Eh.

df_cols = ["VAR", "CNN", "Run"]

funcs = [main_base.main, main_carryon.main]

for file_path in data_files:
	import classes # May need to put this here to ensure counts etc. are reset - TEST THIS
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
	# If labels=True, then we can calc ARI, otherwise we can't, correct this!

	# Remove labels if present and create data_dict
	data, data_dict = classes.createDatasetGarza(data)

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
	print("Precomputation done!")

	HV_ref = None

	for index_d, delta in enumerate(delta_vals):
		# Create tuple of arguments
		args = data, data_dict, delta, HV_ref, argsortdists, nn_rankings, mst_genotype, int_links_indices, L, num_indivs

		###### Arguments to add? ######
		# Amount we reduce delta by
		# Number of generations

		# Initialise df
		# df = pd.DataFrame(columns=df_cols)

		final_obj_values = np.zeros((num_indivs*num_runs,len(df_cols)))

		##### What numbers do I want and where???? #####

		for run in range(num_runs):
			random.seed(seeds[run])

			print("HV ref:", HV_ref)

			if HV_ref == None:
				first_run = True

			else:
				first_run = False

			for func in funcs:
				print("Run",run,"with ",func.__name__)
				start_time = time.time()
				pop, logbook, _, _, HV, ea_time, final_pop_metrics, HV_ref_temp = func(*args)
				end_time = time.time()
				print("Run "+str(run)+" for d="+str(delta)+" complete (Took",end_time-start_time,"seconds)\n")
				if first_run:
					HV_ref = HV_ref_temp

			###### Put the above into a loop over strategies (functions)

			###### Create a folder for graphs, and save the graphs we make into that
			###### Best to do that here and just have the graph func return a graph
				# Easier to aggregate here
				# Easier to save graphs for individual runs from within the main func, or aggregate over a single run with multiple funcs here


			# print(final_pop_metrics,"\n")

			# Check logbook to see how the form looks, and if we can manipulate it into a dataframe
			# May want to just loop through the pop and get the fitness values
			# And add the run number to the 3rd column

			# This is overwriting
			ind = num_indivs*run

			final_obj_values[ind:ind+num_indivs,0:3] = [indiv.fitness.values+(run+1,) for indiv in pop]

		# Modify the below for specific dataset folder
		# np.savetxt(results_path+classes.Dataset.data_name[:-15]+"_eaf_"+str(delta)+".csv", arr, delimiter=" ")