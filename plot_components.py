import pdb
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import run_mock
from classes import Dataset, MOCKGenotype, PartialClust

def component_labels(comp_dict):
    labels = np.empty(Dataset.num_examples)
    labels[:] = np.nan

    for key, clust_obj in comp_dict.items():
        labels[clust_obj.members] = key

    assert not np.any(np.isnan(labels))
    assert len(np.unique(labels)) == len(comp_dict)

    return labels.astype(int)

def get_components(kwargs, delta_val):
    # MOCKGenotype.delta_val = delta_val
    MOCKGenotype.calc_delta(delta_val)
    kwargs['delta_val'] = MOCKGenotype.delta_val
    print(f"Using sr{delta_val} gives delta={MOCKGenotype.delta_val}")

    # Setup some of the variables for the genotype
    MOCKGenotype.setup_genotype_vars()

    PartialClust.partial_clusts(kwargs["data"], kwargs["data_dict"], kwargs["argsortdists"], kwargs["L"])
    MOCKGenotype.calc_reduced_clusts(kwargs["data_dict"])
    # pdb.set_trace()
    labels = component_labels(PartialClust.comp_dict)
    return labels

def save_components(labels, delta_val):
    save_name = f"component_labels_d{delta_val}.csv"
    np.savetxt(save_name, labels, delimiter=",", fmt="%i")
    return save_name

def plot_components(data, labels):
    fig, ax = plt.subplots()

    data = np.hstack((data, labels[:, None]))
    print(data.shape)
    df = pd.DataFrame(data=data, columns=["x", "y", "comp_label"])
    print(df.head())
    df.plot.scatter("x", "y", c="comp_label", colormap="jet", alpha=0.5, s=5)
    plt.show()

def load_data(file_path):
    data = np.genfromtxt(file_path, delimiter="\t", skip_header=4)
    return data[:, :-1]

"""
Need to transfer over the python environment
Then need to look at the plotting funcs that I have used and transfer that if need be
Then just create the graphs needed for paper 1 and text
Then add the figure 1 from paper 2
Then time it roughly to see if it's capable to do both
    With a script
"""

if __name__ == "__main__":
    calc_components = True
    delta_vals = [1, 5, 10, 50, 100, 500, 1000, 10000]
    
    f_paths, res_folder = run_mock.load_data(
        use_real_data=True,
        synth_data_subset="*20_10_3*",
        real_data_subset="*UKC5*"
    )
    data_path = f_paths[1]

    label_list = []

    if calc_components:
        kwargs = run_mock.prepare_data(data_path)
    
        for delta_val in delta_vals:
            labels = get_components(kwargs, delta_val)
            fname = save_components(labels, delta_val)

            # plot_components(kwargs['data'], labels)

    else:
        data = load_data(data_path)

        for delta_val in delta_vals:
            fname = glob.glob(f"*component*{delta_val}.csv")[0]

            labels = np.loadtxt(fname)

            plot_components(data, labels)
            
