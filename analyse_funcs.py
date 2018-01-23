import pandas as pd
import glob
import numpy as np
import os
import fnmatch

import matplotlib.pyplot as plt
from scipy.stats import wilcoxon


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

def saveARIs(artif_folder, method, metric="ari"):
	folders = glob.glob(artif_folder+os.sep+"*_9", recursive=True)

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


	for i, data in enumerate(data_metric_list):
		# print(data, stratname_ref[i])
		if "base" in stratname_ref[i]:
			fname = results_path + os.sep + "artif-allARI-"+stratname_ref[i]+".csv"
			np.savetxt(fname, data, delimiter=",")
		else:
			fname = results_path + os.sep + "artif-allARI-"+stratname_ref[i]+"-"+method+".csv"
			np.savetxt(fname, data, delimiter=",")



def ARIWilcoxon(results_path, strat1, strat2, method1, method2):

	if "base" in strat1:
		data1_fname = glob.glob(results_path+os.sep+"*"+strat1+"*")[0]
	else:
		data1_fname = glob.glob(results_path+os.sep+"*"+strat1+"*"+method1+"*")[0]

	if "base" in strat2:
		data2_fname = glob.glob(results_path+os.sep+"*"+strat2+"*")[0]
	else:
		data2_fname = glob.glob(results_path+os.sep+"*"+strat2+"*"+method2+"*")[0]

	print(data1_fname)
	print(data2_fname)

	data1 = np.loadtxt(data1_fname, delimiter=",")
	data2 = np.loadtxt(data2_fname, delimiter=",")

	# fig, axs = plt.subplots(2, 1, sharex=True, tight_layout=True)
	# axs[0].hist(data1, bins=200)
	# axs[1].hist(data2, bins=200)
	# plt.show()

	sum_ranks, p_val = wilcoxon(data1, data2, zero_method='wilcox')

	print("Comparing strategies:",strat1,"and",strat2)
	print("Using methods:", method1, "and", method2)
	print("Sum Ranks:",sum_ranks)
	print("P-Value:", p_val,"\n")
	print("Medians:", strat1, np.median(data1), strat2, np.median(data2))
	print("Means:", strat1, np.mean(data1), strat2, np.mean(data2))
	print("Mins:", strat1, np.min(data1), strat2, np.min(data2))


if __name__ == '__main__':
	basepath = os.getcwd()
	results_path = os.path.join(basepath, "results")
	artif_folder = os.path.join(results_path, "artif")

	methods = ["random", "hv"]
	strategies = ["base-sr1", "base-sr5", "carryon", "fairmut", "hypermutall", "hypermutspec", "reinit"]

	# for method in methods:
	# 	saveARIs(artif_folder, method)

	ARIWilcoxon(results_path, "base-sr5", "reinit", "random","random")
	# ARIWilcoxon(results_path, strategies[-1], strategies[-1], methods[0], methods[1])
