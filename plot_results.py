import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import main
import random

import adaptive_fairmut

seeds = [1,5,7,10]

#### Might need to put the DEAP stuff here as if we import multiple scripts we define an indiv multple times

# Create a list of random numbers here
# Then loop through them and set each one as the seed for a given run

### Rename to generate results
# Plot results in a different script

# delta_vals = [95]*20
delta_vals = [90]*3
# delta_vals = [i for i in range(70,100,5) for _ in range(5)]
print(list(delta_vals))

base_path = os.getcwd()

## Set data_folder
# data_folder = base_path+"/data/data_handl/"
data_folder = base_path+"/data/iris/"

## Set results folder
results_folder = base_path+"/results/"

# Check data folder exists
if not os.path.isdir(data_folder):
	raise SystemExit("No data folder found, please check a valid dataset name has been specified")

## Select dataset
# data_name = "ellipsoid-50d20c-1.dat"
# data_name = "10d-20c-no0.dat"
# data_name = "10d-40c-no9.dat"
# data_name = "ellipsoid-100d40c-2.dat"
data_name = "iris.csv"

data_path = data_folder+data_name

if len(data_name.split("."))>2:
	print("Warning: Data file name has more than 1 period (.), resulting names may be weird")

# get a clean name for the data without the extension
data_name_clean = data_name.split(".")[0]

# Initialise dataframes
h_vols = pd.DataFrame()
var_vals = pd.DataFrame()
cnn_vals = pd.DataFrame()
ea_times = []

# Keep same point consistent throughout each gen
# max_conn cannot be exceeded (but is different for different delta vals)
# We use double the worst VAR value in initial pop
# We'll need to pickle this so that it is the same for every run, so need a try except statement
# HV_ref = [np.max(VAR_init)*2, max_conn+1]
# Check values below over range of delta values so we get a point that works for all

### This needs to be automated
## Consider using rounding to ensure small differences in initialistion don't matter
## just order of magnitude (perhaps a dictionary of HV ref points for delta values)
# HV_ref = [20.0, 200.0] # Iris
# HV_ref = [400.0, 2000.0] # Digits
# HV_ref = [1000.0, 6000.0] # 10d-40c-no0.dat
# HV_ref = [5.0, 3800.0] # ellipsoid.50d20c.1.dat
HV_ref = [1000.0, 4000.0] # 10d-20c-no0.dat

# HV_ref = None # Dynamic ref

for run, delta in enumerate(delta_vals):
	# pop, logbook, _, _, HV, ea_time = main.main(data_path, delta, HV_ref)
	random.seed(seeds[run]) # Works!
	# pop, logbook, _, _, HV, ea_time, HV_ref, final_pop_metrics = main.main(data_path, delta, HV_ref) # Dynamic ref
	pop, logbook, _, _, HV, ea_time, HV_ref, final_pop_metrics = adaptive_fairmut.main(data_path, delta, HV_ref)
	h_vols["Run "+str(run)+": "+str(delta)] = HV

	ea_times.append(ea_time)

	VARs = []
	CNNs = []
	for indiv in pop:
		VARs.append(indiv.fitness.values[0])
		CNNs.append(indiv.fitness.values[1])

	var_vals[delta] = VARs
	cnn_vals[delta] = CNNs

# print(h_vols)
# print(ea_times)
# print(var_vals)
# print(cnn_vals)

# # To pickle the dataframes
# h_vols.to_pickle(data_name+'_hv.pkl')
# var_vals.to_pickle(data_name+'_var.pkl')
# cnn_vals.to_pickle(data_name+'_cnn.pkl')

# Save in results folder (and create if not present)
try:
	h_vols.to_csv(results_folder+data_name_clean+"_results_hv.csv", index=False)
except FileNotFoundError:
	os.mkdir(results_folder)
	h_vols.to_csv(results_folder+data_name_clean+"_results_hv.csv", index=False)

# Pickle the EA times
filename = results_folder+data_name_clean+"_ea_times.txt"
with open(filename, "wb") as fp:
	pickle.dump(ea_times, fp)


# # Loop through files to build the dataframe with all results
# for file in os.listdir(data_folder):
# 	filename = os.fsdecode(file) # Not really needed, file is the same string

# 	if filename.endswith("_hv.csv"):
# 		# print(filename)
# 		# runval will be index 2 when split by _
# 		col_label = filename.split("_")[2]
# 		continue

# 	elif filename.endswith("_finalpop.csv"):
# 		# print(filename)
# 		col_label = filename.split("_")[2]
# 		continue

# 	# # Redundant
# 	# else:
# 	# 	continue

# hv = pd.read_csv(data_folder+"/"+data_name+"_delta"+str(delta_val)+"_hv.csv")

# Plot results

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.scatter(range(0,NUM_GEN), h_vols, s=3)

# plt.show()

#### See below
### From: https://stackoverflow.com/questions/20906474/import-multiple-csv-files-into-pandas-and-concatenate-into-one-dataframe
# df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
# Also, one should perhaps use os.path.join(path, "*.csv") instead of path + "/*.csv", which makes it OS independent.

# Therefore, may be better to split result sinto separate folders (for hv & final_pop)
# To facilitate the approach above


### To ensure absolute consistency of HV value we need to calculate the ref
### Then maybe store it in a dictionary for that dataset (as the key) so we can retain it
# Current MOCK gets around this by normalising, then using the ref [1.01,1.01]
# Not sure if we can do this though because of how are we are using HV
# For reporting the HV of the final pop we can do this, but to compare HV between values - not sure