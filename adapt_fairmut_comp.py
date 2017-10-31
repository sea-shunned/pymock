import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import numpy as np

base_path = os.getcwd()
results_path = base_path+"/results/fair_mut/"

hv_files = glob.glob(results_path+'*.csv')
hv_files = glob.glob(results_path+'*d90.csv')
# hv_files.extend(glob.glob(results_path+'*d99.csv'))

# print(hv_files)

# data_dict = {}

hv_files = sorted(hv_files)

fig, ax = plt.subplots()

for file in hv_files:
	data = pd.read_csv(file)
	# data = data.apply(np.log)
	clean_name = file.split("/")[-1].split(".")[0]
	legend_name = clean_name.split("_")[1]+" - "+clean_name.split("_")[2]

	if "d50" in clean_name:
		color = "r"
	elif "d70" in clean_name:
		color = "b"
	elif "d90" in clean_name:
		color = "g"
	else:
		color = "y"

	if "normal" in clean_name:
		linestyle = "solid"

	if "adapt" in clean_name:
		linestyle = "dotted"

	if "fairmut" in clean_name:
		linestyle = "dashed"


	linestyles = ["solid", "dash", "dotted"]

	# Normalise the data between 0 & 1
	# data_min = data.min().min()
	# data_max = data.max().max()
	# data_scaled = (data - data_min) / (data_max - data_min)

	data_avg = data.mean(axis=1)

	plt.plot(range(0,len(data)), data_avg, color=color, linewidth=2.5, linestyle=linestyle, label=str(legend_name))


ax.set_xlabel("Generation")
ax.set_ylabel("Hypervolume")
ax.set_title("Comparison of Approaches (averaged over 20 runs)")
ax.legend()
plt.show()

# hatch	[‘/’ | ‘\’ | ‘|’ | ‘-‘ | ‘+’ | ‘x’ | ‘o’ | ‘O’ | ‘.’ | ‘*’]


def eaTimes():
	import pickle
	files = glob.glob(results_path+'*.txt')
	# print(files)

	files = sorted(files)

	fig, ax = plt.subplots()

	width = 0.3

	for file in files:
		clean_name = file.split("/")[-1].split(".")[0]
		legend_name = clean_name.split("_")[1]+" - "+clean_name.split("_")[2]
		with open(file, 'rb') as fp:
			ea_times = pickle.load(fp)
			# print(clean_name)
			# print(ea_times)

		height_val = np.mean(ea_times)
		# print(height_val)

		if "d50" in clean_name:
			hatch = None
			counter = 0

		if "d70" in clean_name:
			hatch = "///"
			counter = 1

		if "d90" in clean_name:
			hatch = "++"
			counter = 2

		if "normal" in clean_name:
			# color = "g"
			color = "#f20253"
			step = 1

		if "adapt" in clean_name:
			color = "#2185c5"
			step = 2

		if "fairmut" in clean_name:
			color = "#ff9715"
			step = 3

		plt.bar(counter+(width*step), height_val, width, color=color, hatch=hatch, edgecolor='black', label=legend_name)

	ax.set_xticks([0.6,1.6,2.6])
	ax.set_xticklabels(["50","70","90"])
	ax.set_xlabel("Delta Value")
	ax.set_ylabel("Time Taken (secs)")
	ax.set_title("Time Taken for Different Strategies and Different Delta Values")
	ax.legend()
	plt.show()

# eaTimes()