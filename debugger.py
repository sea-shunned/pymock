import os
import pandas as pd
import csv
# import matplotlib.pyplot as plt
import numpy as np
import pickle
import glob
import classes
import precompute
import time

import random

import main_base
import main_carryon
import main_carryon_old

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

file_path = data_files[0]
# print(file_path)

num_runs = 2
# seeds = [random.randint(0,10000)*i for i in range(num_runs)]
seeds = [11, 1000]
# Pickle (save) the seeds here

# Set range of delta values to test for each file
# delta_vals = [i for i in range(90,100,5) for _ in range(num_runs)]
delta_vals = [i for i in range(90,97,5)]
# delta_vals = [90]

print("Delta values to test:", delta_vals)
print("Number of runs per delta value:", num_runs)
print("Number of datasets:", len(data_files))
print("Number of total MOCK Runs:", len(data_files)*len(delta_vals)*num_runs,"\n")

## Below will be kept in the actual main function
## But just put here for a checklist
# Check that 100 indivs and 100 generations is OK
L = 10
num_indivs = 100
# Check crossover probability
# Or do I want to put this stuff here? Eh.

df_cols = ["VAR", "CNN", "Run"]

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
print("Precomputation done!\n")
# Which of the above do I need to pass?
# argsortdists, nn_rankings, mst_genotype, int_links_indices



HV_ref = None
HV_vals = pd.DataFrame(columns=delta_vals)

for index_d, delta in enumerate(delta_vals):

	final_obj_values = np.zeros((num_indivs*num_runs,len(df_cols)))

	for run in range(num_runs):
		random.seed(seeds[run])

		print("HV ref:", HV_ref)

		if HV_ref == None:
			first_run = True

		else:
			first_run = False

		print("Run",run,"with main_base")
		start_time = time.time()
		pop, logbook, _, _, HV, ea_time, final_pop_metrics, HV_ref_temp = main_base.main(data, data_dict, delta, HV_ref, argsortdists, nn_rankings, mst_genotype, int_links_indices, L, num_indivs)
		end_time = time.time()
		print("Run "+str(run)+" for d="+str(delta)+" complete (Took",end_time-start_time,"seconds)\n")

		# pop, logbook, _, _, HV, ea_time, final_pop_metrics, HV_ref_temp = main_carryon_old.main(data, data_dict, delta, HV_ref, argsortdists, nn_rankings, mst_genotype, int_links_indices, L, num_indivs)
		# print("\n")
		
		print("Run",run,"with main_carryon")
		start_time = time.time()
		pop, logbook, _, _, HV, ea_time, final_pop_metrics, HV_ref_temp = main_carryon.main(data, data_dict, delta, HV_ref, argsortdists, nn_rankings, mst_genotype, int_links_indices, L, num_indivs)
		end_time = time.time()
		print("Run "+str(run)+" for d="+str(delta)+" complete (Took",end_time-start_time,"seconds)\n")
		
		if first_run:
			HV_ref = HV_ref_temp

		# print(final_pop_metrics,"\n")

		# print(final_pop_metrics)
		# print(np.mean(final_pop_metrics["Num Clusters"]))

		# Check logbook to see how the form looks, and if we can manipulate it into a dataframe
		# May want to just loop through the pop and get the fitness values
		# And add the run number to the 3rd column

		ind = num_indivs*run

		# Add 1 to run so it's 1-based, may help with the R EAF stuff
		final_obj_values[ind:ind+num_indivs,0:3] = [indiv.fitness.values+(run+1,) for indiv in pop]

	# Can modify the below to take the mean HV vals if we do multiple runs and want to
	HV_vals[delta] = HV

	# Modify the below for specific dataset folder
	# np.savetxt(results_folder+classes.Dataset.data_name+"_eaf_"+str(delta)+".csv", final_obj_values, delimiter=" ")

# Save anything that was aggregated over the delta values
# print(HV_vals)

# HV_vals.to_csv(results_folder+classes.Dataset.data_name+"_hv.csv", index=False)