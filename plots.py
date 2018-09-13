import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# try to construct this using the pathlib module

def get_folders(folder_str):
    results_folder = Path.cwd() / "results"
    
    # for folder in results_folder.glob(folder_str+"*"):
    #     print(folder)

    print(results_folder.glob(folder_str / "*"))

def get_fpaths(filters):
    # Get some file paths
    # Could have string to select subset
    # also arg that specifies folders
    # do we want a list (or lis tof lists, where each sublist is a folder)?
    # or just handle single folders
    # pass

    results_folder = Path.cwd() / "results"

    if not results_folder.is_dir():
        raise ValueError(f"{results_folder} is not a directory!")

    glob_str = "/".join(filters)
    f_paths = [str(fname) for fname in results_folder.glob(glob_str)]

    print(f_paths)

def plot_boxplot(data, ax, x_label, y_label, **kwargs):
    # can use **kwargs for mpl props
    # May take some fiddling
    pass

    # medianprops = dict(linewidth=2, color='midnightblue')

    bxplot = ax.boxplot(data, **kwargs)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    return ax

def save_graph(fig_obj, save_path):
    # plt.savefig(save_path, etc.)
    pass

def norm_data(data):
    # function to normalise data
    # maybe need this
    # prev used when normalising time taken data
    pass

def stats_test(data_1, data_2, test_name="wilcoxon"):
    try:
        stats_func = getattr(stats, test_name)
    # sum_ranks, p_val = wilxocon(data_1, data_2)
    except AttributeError:
        raise(f"Cannot find {test_name} in scipy.stats!")

    if test_name == "wilcoxon":
        sum_ranks, p_val = stats_func(data_1, data_2)
        return sum_ranks, p_val
    else:
        raise ValueError(f"Whoopsie! Behaviour not yet implemented...")

    # Asking for a general func may be too much - the below might even fail with some funcs
    # scipy.stats don't have unified function behaviour (obvs)
    # return stats_func(data_1, data_2)

def stats_colours(data):
    colours = []

    for data_set in data:
        pass


# would having one big container be useful?
# like a dictionary where the keys are the different sets of results
# then have sub_dicts with name, if it's the reference, actual data etc.

results = {

}


boxplot_kwargs = {
    'medianprops': {
        'linewidth': 2,
        'color': 'midnightblue'
    },
    'patch_artist': True
}


if __name__ == '__main__':
    # get_folders("test")
    get_fpaths(["test","centroid","*ari*"])