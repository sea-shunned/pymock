import pandas as pd
import glob
import numpy as np
import os
import fnmatch

def stdError(folder_path):
	# Create a dataframe with the average, std dev, and std error for each delta value

	data_types = ["ari", "fitness", "hv", "numclusts"]

	strategies = ["main_base", "main_carryon", "main_hypermutspec", "main_hypermutall", "main_reinit", "main_fairmut"]

	for dat in data_types:
		df_results = pd.DataFrame()

		files = glob.glob(folder_path+"*-"+dat+"-*")
		
			# print(file)
		# print("\n")
		for strategy in strategies:
			for file in files:
				if strategy in file:
					delta = file.split("-")[-1].split(".")[0]
					data = np.loadtxt(file, delimiter=",")
					# print(data)
					# print(strategy, delta, dat)
					# print(file,"\n")


	####	####	####	####	####	####

	for strategy in strategies:

		files = glob.glob(folder_path+"*-")



	# pass

def statTest():
	pass



if __name__ == '__main__':
	basepath = os.getcwd()

	folder_path = basepath+"/results/tevc_20_60/"

	stdError(folder_path)