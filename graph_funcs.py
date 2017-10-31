import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt

def normEAF(folder_path):
	files = glob.glob(folder_path+'*.csv*')

	# print(files)

	# Get the unique dataset names
	# uniq_data = set(['_'.join(file.split("/")[-1].split("_")[:-4]) for file in files])

	# Set to 0 or infinity so that we ensure we update
	max_val_var = 0
	min_val_var = np.inf

	max_val_cnn = 0
	min_val_cnn = np.inf # always 0 due to MST solution, check with Julia and remove

	for file in files:
		data = np.genfromtxt(file, delimiter=" ")
		data = data[:,:2]
		# print(data)
		
		curr_max_val_var = np.max(data[:,0])
		curr_min_val_var = np.min(data[:,0])

		curr_max_val_cnn = np.max(data[:,1])
		curr_min_val_cnn = np.min(data[:,1])

		# Remember signs!
		if max_val_var < curr_max_val_var:
			max_val_var = curr_max_val_var

		if min_val_var > curr_min_val_var:
			min_val_var = curr_min_val_var

		if max_val_cnn < curr_max_val_cnn:
			max_val_cnn = curr_max_val_cnn

		if min_val_cnn > curr_min_val_cnn:
			min_val_cnn = curr_min_val_cnn

	# print(max_val_var, max_val_cnn)
	# print(min_val_var, min_val_cnn)

	denom_var = max_val_var - min_val_var
	denom_cnn = max_val_cnn - min_val_cnn

	for file in files:
		# Normalise the data
		data = np.genfromtxt(file, delimiter=" ")
		data[:,0] = (data[:,0] - min_val_var)/denom_var
		data[:,1] = (data[:,1] - min_val_cnn)/denom_cnn

		# print(data)

		name = file.split(".")[0]+"_norm.csv"
		# print(file.split(".")[0])
		print(name)
		np.savetxt(name, data, delimiter = " ")
		# np.savetxt(results_basepath+classes.Dataset.data_name+"_eaf_"+str(delta)+".csv", arr, delimiter=" ")

	### From precompute normdist func
	# max_val = np.max(data)
	# min_val = np.min(data)
	# denom = max_val - min_val
	# for row, val in enumerate(distarray):
	# 	distarray[row] = (val - min_val)/denom


def plotObjectives(csv_path):
	data = np.genfromtxt(csv_path, delimiter=" ")

	if data.shape[1] == 3:
		data = np.delete(data, -1, 1)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	plt.scatter(data[0:99,0], data[0:99,1], s=2, marker="x")
	plt.show()

# https://matplotlib.org/examples/color/color_cycle_demo.html
def plotHV(csv_path):
	df_hv = pd.read_csv(csv_path)

	fig = plt.figure()
	ax = fig.add_subplot(111)

	# is below a list or a generator?
	colors = plt.cm.rainbow(np.linspace(0, 5, len(df_hv)))
	# colors = iter(cm.rainbow(np.linspace(0, 1, len(df_hv))))

	for index, column in enumerate(df_hv):
		# print(len(df_hv[column]))
		plt.plot(range(0,100), df_hv[column], color=colors[index], label=list(df_hv)[index]) #next(colors)
	plt.legend()
	plt.show()


if __name__ == '__main__':
	results_path = "/home/cshand/Documents/PhD_MOCK/adaptive_delta/results/"
	# /home/cshand/Dropbox/Computer Science PhD/PhD_Y2/MOEA_Clustering/mock/adaptive_delta/results/eafs

	folders = glob.glob(results_path+'*/')
	folders = folders[:1]
	# print(folders)

	# for folder in folders:
	# 	normEAF(folder)

	# csv_path = "/home/cshand/Documents/PhD_MOCK/adaptive_delta/results/tevc_50_100_2/tevc_50_100_2_labels_headers_eaf_30.csv"
	# plotObjectives(csv_path)

	csv_path = "/home/cshand/Documents/PhD_MOCK/adaptive_delta/results/tevc_20_10_6_labels_headers_hv.csv"
	plotHV(csv_path)