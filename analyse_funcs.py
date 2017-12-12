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

	checker = input("Have you checked if you're overwriting files from a previous experiment? (Answer y to continue) ")

	if not (checker == "y" or checker == "Y"):
		raise SystemExit


	for strategy in strategies:

		for dat in data_types:

			df_results = pd.DataFrame()

			for folder in results_folders:
				data_name = folder.split("/")[-2]


				files = glob.glob(folder+"*"+strategy+"*"+dat+"*")
				# print(files)

				for file in files:
					delta = file.split("-")[-1].split(".")[0]
				

					data = np.loadtxt(file, delimiter=',')

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

					df_results.loc[data_name,"mean_d"+str(delta)] = mean_val
					df_results.loc[data_name,"stddev_d"+str(delta)] = std_dev
					df_results.loc[data_name,"stderr_d"+str(delta)] = std_err

					if dat != "hv":
						df_results.loc[data_name,"max_d"+str(delta)] = max_val
						df_results.loc[data_name,"min_d"+str(delta)] = min_val

			print(df_results)
		# print(folder,"\n")

			


			
			save_name = aggregate_folder+strategy+"-"+dat+"-initial.csv" 

			# Save dataframe
			df_results.to_csv(save_name,sep=",",header=True,index=True)
				





	# pass

###### Have separate functions for the calcualtions?

def statTest():
	pass



if __name__ == '__main__':
	basepath = os.getcwd()

	# folder_path = basepath+"/results/tevc_20_60/"

	calcAggregates()