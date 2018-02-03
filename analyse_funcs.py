import pandas as pd
import glob
import numpy as np
import os
import fnmatch

import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
from scipy.stats import friedmanchisquare


def calcAggregates():
	# Create a dataframe with the average, std dev, and std error for each delta value

	basepath = os.getcwd()
	aggregate_folder = basepath+"/results/aggregates/"
	if not os.path.isdir(aggregate_folder):
		os.makedirs(aggregate_folder)

	data_types = ["ari", "hv", "numclusts"]
	# fitness will be looked at in R for EAFs

	# Could take this as input but probably constant or just modifiable 
	strategies = ["main_base", "main_carryon", "main_hypermutspec", "main_hypermutall", "main_reinit", "main_fairmut"]

	results_folders = glob.glob(basepath+"/results/*/")

	# Remove aggregates folder to avoid looping through it
	results_folders.remove(aggregate_folder)
	results_folders.remove(basepath+"/results/graphs/")

	checker = input("Have you checked if you're overwriting files from a previous experiment? (Answer y to continue) ")

	if not (checker == "y" or checker == "Y"):
		raise SystemExit


	for folder in results_folders:
		data_name = folder.split("/")[-2]

		for dat in data_types:

			df_results = pd.DataFrame()

			for strategy in strategies:

				files = glob.glob(folder+"*"+strategy+"*"+dat+"*")
				# print(files)

				for file in files:
					delta = file.split("-")[-1].split(".")[0]

					data = np.loadtxt(file, delimiter=',')

					# Assignment redundant, just clearer to see

					# HV we just want the average etc. from the final population
					if dat == "hv":
						mean_val = np.mean(data[-1,:])
						std_dev = np.std(data[-1,:], ddof=0)

					else:	
						# This takes the mean across all of the data
						# So for all individuals, over all runs
						mean_val = np.mean(data)
						std_dev = np.std(data, ddof=0)

						max_val = np.max(data)
						min_val = np.min(data)
						
					n = data.shape[1]
					std_err = std_dev / n

					# print(mean_val)

					df_results.loc[strategy,"mean_d"+str(delta)] = mean_val
					df_results.loc[strategy,"stddev_d"+str(delta)] = std_dev
					df_results.loc[strategy,"stderr_d"+str(delta)] = std_err

					if dat != "hv":
						df_results.loc[strategy,"max_d"+str(delta)] = max_val
						df_results.loc[strategy,"min_d"+str(delta)] = min_val

			print(df_results)
		# print(folder,"\n")
			
			save_name = aggregate_folder+data_name+"-"+dat+"-initial.csv" 

			# Save dataframe
			df_results.to_csv(save_name,sep=",",header=True,index=True)

###### Have separate functions for the calcualtions?

def statTest():
	pass

def aggregHV():
	# Aggregate the HV values for each generation

	basepath = os.getcwd()
	aggregate_folder = basepath+"/results/aggregates/"
	if not os.path.isdir(aggregate_folder):
		os.makedirs(aggregate_folder)

	data_types = ["hv"]
	# fitness will be looked at in R for EAFs

	# Could take this as input but probably constant or just modifiable 
	strategies = ["main_base", "main_carryon", "main_hypermutspec", "main_hypermutall", "main_reinit", "main_fairmut"]

	results_folders = glob.glob(basepath+"/results/*/")

	# Remove aggregates folder to avoid looping through it
	results_folders.remove(aggregate_folder)
	results_folders.remove(basepath+"/results/graphs/")

	checker = input("Have you checked if you're overwriting files from a previous experiment? (Answer y to continue) ")

	if not (checker == "y" or checker == "Y"):
		raise SystemExit


	for folder in results_folders:
		data_name = folder.split("/")[-2]

		for dat in data_types:

			df_results = pd.DataFrame()

			for strategy in strategies:

				files = glob.glob(folder+"*"+strategy+"*"+dat+"*")
				# print(files)

				for file in files:
					delta = file.split("-")[-1].split(".")[0]

					data = np.loadtxt(file, delimiter=',')
					# Assignment redundant, just clearer to see
					means = np.mean(data, axis=1)
					std_devs = np.std(data, axis=1, ddof=0)
					std_errs = std_devs / data.shape[1]

					df_results.loc[:,"mean_"+strategy+"_d"+str(delta)] = means
					df_results.loc[:,"stddev_"+strategy+"_d"+str(delta)] = std_devs
					df_results.loc[:,"stderr_"+strategy+"_d"+str(delta)] = std_errs

			# print(df_results)
			save_name = aggregate_folder+data_name+"-HVgens-initial.csv" 

			df_results.to_csv(save_name,sep=",",header=True,index=False)

def saveARIs(artif_folder, method, dataname="*_9", metric="ari"):
	folders = glob.glob(artif_folder+os.sep+dataname, recursive=True)

	# Have the strategies here in a defined order, then just check that the one extracted from the filename matches to ensure consistency
	stratname_ref = ["base-sr1", "base-sr5", "carryon", "fairmut", "hypermutall", "hypermutspec", "reinit"]

	# Lists to aggregate the data over all datasets
	data_metric_list = []

	for num_dataset, dataset_folder in enumerate(folders):
		metric_files = glob.glob(dataset_folder+os.sep+"*base*"+metric+"*")
		metric_files.extend(glob.glob(dataset_folder+os.sep+"*"+metric+"*"+method+"*"))

		metric_files = sorted(metric_files, reverse=False)

		strat_names = []

		# print(metric_files,"\n")

		# Extract data_name
		data_name = dataset_folder.split(os.sep)[-1]

		for index, file in enumerate(metric_files):
			# print(file)
			
			data_metric = np.max(np.loadtxt(file, delimiter=","),axis=0)

			if "base" in file:
				# strat_names.append("-".join([file.split(os.sep)[-1].split("-")[1].split("_")[-1],
				# 	file.split(os.sep)[-1].split("-")[-1].split(".")[0][:3]]))

				strat_names.append("-".join([file.split(os.sep)[-1].split("-")[1],file.split(os.sep)[-1].split("-")[3][:-4]]))

				# print("-".join([file.split(os.sep)[-1].split("-")[1].split("_")[-1],
				# 	file.split(os.sep)[-1].split("-")[-1].split(".")[0]]))
				# print("-".join([file.split(os.sep)[-1].split("-")[1],file.split(os.sep)[-1].split("-")[3]]))

			else:
				strat_names.append(file.split(os.sep)[-1].split("-")[1].split("_")[-1])

			# Show order of strategies
			# print(strat_names[-1], index, stratname_ref[index])

			assert strat_names[-1] == stratname_ref[index], "Strat name difference "+strat_names[-1]+" "+stratname_ref[index]

			# Create initial arrays for the first dataset, then append afterwards
			# The boxplot command can then handle everything
			# We should have just a single array for each of the strategies

			# strat_index = stratname_ref.index(strat_names[-1])
			# print(strat_names[-1], index, strat_index)

			# It could be useful to use stratname_ref.index(strat_names[-1]) to avoid enumerate for loop issue with empty datasets (though that shouldn't be a problem for the _9_ datasets)

			if num_dataset == 0:
				data_metric_list.append(data_metric)

			else:
				data_metric_list[index] = np.append(data_metric_list[index], data_metric)

	if dataname == "*UKC*":
		datatype = "real"
	else:
		datatype = "synth"

	for i, data in enumerate(data_metric_list):
		if "base" in stratname_ref[i]:
			fname = results_path + os.sep + "artif-allARI-"+datatype+"-"+stratname_ref[i]+".csv"
			np.savetxt(fname, data, delimiter=",")
		else:
			fname = results_path + os.sep + "artif-allARI-"+datatype+"-"+stratname_ref[i]+"-"+method+".csv"
			np.savetxt(fname, data, delimiter=",")


def ARIWilcoxon(results_path, strat1, strat2, method1, method2):
	print(glob.glob(results_path+os.sep+"*"+strat1+"*"))
	if "base" in strat1:
		data1_fname = glob.glob(results_path+os.sep+"*"+strat1+"*")[0]
	else:
		data1_fname = glob.glob(results_path+os.sep+"*"+strat1+"*"+method1+"*")[0]

	if "base" in strat2:
		data2_fname = glob.glob(results_path+os.sep+"*"+strat2+"*")[0]
	else:
		data2_fname = glob.glob(results_path+os.sep+"*"+strat2+"*"+method2+"*")[0]

	# print(data1_fname)
	# print(data2_fname)

	data1 = np.loadtxt(data1_fname, delimiter=",")
	data2 = np.loadtxt(data2_fname, delimiter=",")

	# fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True)
	# axs[0].hist(data1, bins=200)
	# axs[1].hist(data2, bins=200)
	# plt.show()

	sum_ranks, p_val = wilcoxon(data1, data2, zero_method='wilcox')

	print("\nComparing strategies:",strat1,"and",strat2)
	print("Using methods:", method1, "and", method2)
	print("Sum Ranks:",sum_ranks)
	print("P-Value:", p_val,"\n")

	# sum_ranks, p_val = wilcoxon(data1, data2, zero_method='pratt')
	# print("Sum Ranks:",sum_ranks)
	# print("P-Value:", p_val,"\n")

	print("Medians:", strat1, np.median(data1), strat2, np.median(data2))
	print("Means:", strat1, np.mean(data1), strat2, np.mean(data2))
	print("Mins:", strat1, np.min(data1), strat2, np.min(data2))

def TimeDiffs(artif_folder, method, dataname="*_9*"):
	folders = glob.glob(artif_folder+os.sep+dataname, recursive=True)

	stratname_ref = ["base-sr1", "base-sr5", "carryon", "fairmut", "hypermutall", "hypermutspec", "reinit"]

	# Lists to aggregate the data over all datasets
	data_time_list = []

	# box_colours = 

	# print(folders, len(folders))

	for num_dataset, dataset_folder in enumerate(folders):
		time_files = glob.glob(dataset_folder+os.sep+"*base*time*")
		time_files.extend(glob.glob(dataset_folder+os.sep+"*time*"+method+"*"))

		time_files = sorted(time_files, reverse=False)

		# print(time_files, len(time_files))

		# if time_files == []:
		# 	continue

		strat_names = []

		# Constants for normalising the data between 0 and 1
		min_val = np.inf
		max_val = 0

		# Extract data_name
		data_name = dataset_folder.split(os.sep)[-1]

		for index, file in enumerate(time_files):
			# print(file)
			# print(time_files, len(time_files))
			
			data_time = np.loadtxt(time_files[index], delimiter=',')
			
			if "base" in file:
				# strat_names.append("-".join([file.split(os.sep)[-1].split("-")[1].split("_")[-1],
				# 	file.split(os.sep)[-1].split("-")[-1].split(".")[0][:3]]))
				strat_names.append("-".join([file.split(os.sep)[-1].split("-")[1],file.split(os.sep)[-1].split("-")[3][:-4]]))

				# print("-".join([file.split(os.sep)[-1].split("-")[1].split("_")[-1],
				# 	file.split(os.sep)[-1].split("-")[-1].split(".")[0]]))
				# print("-".join([file.split(os.sep)[-1].split("-")[1],file.split(os.sep)[-1].split("-")[3]]))

			else:
				# print(file.split(os.sep)[-1].split("-")[1].split("_")[-1])
				strat_names.append(file.split(os.sep)[-1].split("-")[1].split("_")[-1])

			# Show order of strategies
			# print(strat_names[-1], index, stratname_ref[index])

			assert strat_names[-1] == stratname_ref[index], "Strat name difference "+strat_names[-1]+" "+stratname_ref[index]

			# Create initial arrays for the first dataset, then append afterwards
			# The boxplot command can then handle everything
			# We should have just a single array for each of the strategies

			# strat_index = stratname_ref.index(strat_names[-1])
			# print(strat_names[-1], index, strat_index)

			# It could be useful to use stratname_ref.index(strat_names[-1]) to avoid enumerate for loop issue with empty datasets (though that shouldn't be a problem for the _9_ datasets)

			if np.max(data_time) > max_val:
				max_val = np.max(data_time)

			if np.min(data_time) < min_val:
				min_val = np.min(data_time)

		denom = max_val - min_val

		for index, file in enumerate(time_files):
			data_time = np.loadtxt(file, delimiter=',')
			data_time = (data_time - min_val)/denom

			if num_dataset == 0:
				data_time_list.append(data_time)

			else:
				# data_time_list[index].append(data_time)
				data_time_list[index] = np.append(data_time_list[index], data_time)

	means = []
	medians = []
	errs = []

	# print(len(data_time_list))
	print(method, dataname)
	for i, times in enumerate(data_time_list):
		means.append(np.mean(times))
		medians.append(np.median(times))
		errs.append(np.std(times, ddof=0)/np.sqrt(times.shape[0]))

		print(strat_names[i], stratname_ref[i])
		print("Mean:", np.mean(times))
		print("Median:", np.median(times))
		print("Std error:", np.std(times, ddof=0)/np.sqrt(times.shape[0]))

	print((means[-1]-means[1])/means[1])
	print((medians[-1]-medians[1])/medians[1],"\n")

def ARIFriedman(results_path, dataset_type="*synth*", method="*interval*"):
	files = glob.glob(results_path+os.sep+dataset_type+"base*")
	files.extend(glob.glob(results_path+os.sep+dataset_type+method))

	assert len(files) == 7, "Don't have 7 files, double check (probably base MOCK issue)"

	data = [np.loadtxt(file, delimiter=",") for file in files]

	print(friedmanchisquare(*data))

if __name__ == '__main__':
	basepath = os.getcwd()
	results_path = os.path.join(basepath, "results")
	artif_folder = os.path.join(results_path, "artif")

	methods = ["random", "interval", "hv"]
	strategies = ["base-sr1", "base-sr5", "carryon", "fairmut", "hypermutall", "hypermutspec", "reinit"]

	# for method in methods:
	# 	saveARIs(artif_folder, method, dataname="*UKC*")

	# ARIWilcoxon(results_path, "base-sr5", "reinit", "interval","interval")
	# ARIWilcoxon(results_path, strategies[-1], strategies[-1], methods[0], methods[1])

	# for method in methods:
	# 	ARIWilcoxon(results_path, "base-sr5", "reinit", method, method)

	# ARIWilcoxon(results_path, "reinit", "reinit", methods[1], methods[2])

	for method in methods:
		TimeDiffs(artif_folder, method)#, dataname="*UKC*")

	# TimeDiffs(artif_folder, method="random", dataname="*UKC*")

	ARIFriedman(results_path, dataset_type="*real*", method="*interval*")