import pandas as pd
import glob
import numpy as np
import os
import fnmatch

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


if __name__ == '__main__':
	basepath = os.getcwd()

	# folder_path = basepath+"/results/tevc_20_60/"

	# calcAggregates()
	aggregHV()