# MOCK in Python
This is Python code for MOCK, Delta-MOCK and Auto-MOCK (still to include Adaptive Delta-MOCK, used in the paper *"Towards an Adaptive Encoding for Evolutionary Data Clustering"*<sup>1</sup>).

<sup>1</sup>Cameron Shand, Richard Allmendinger, Julia Handl, and John Keane. 2018. Towards an Adaptive Encoding for Evolutionary Data Clustering. In GECCO’18: Genetic and Evolutionary Computation Conference, July 15–19, 2018, Kyoto, Japan. ACM, New York, NY, USA, 8 pages. https://doi.org/10.1145/3205455.3205506

For the original Delta-MOCK in high performance C++, please see Mario Garza-Fabre's code [here](https://github.com/garzafabre/Delta-MOCK). This repo is intended for easy development, extensions, and, simple application of Delta-MOCK.

If you are having issues using this code, please don't hesitate to contact the repo owner (cameron.shand (at) manchester.ac.uk). 

## Requirements
The key requirements are:
* Python 3.6+ 
* DEAP 1.2 (probably best to pip install from their github)
* python-igraph (best way is to install through conda-forge)
* Usual python stack (numpy, scipy etc.)

If you are on Linux, a conda environment has been provided. See the section below for install instructions.

## Quick Start
If it is available, install the conda environment, run `conda env create -f mock_env.yml`.

Then, import either of the MOCK, DMOCK or AutoMOCK classes, customise and run.

```
from pymock import AutoMOCK
import pandas as pd

# Read data
df = pd.read_csv('../data/tevc_20_60_9_labels_headers.data', sep='\t', skiprows=4, header=None)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]  # y is only needed for ARI computation

# Load the class and fit
auto_mock = AutoMOCK(k_user=60, num_gens=50)
auto_mock.fit(X, y)

# There are two main datasets as output: results (individuals), hvs (hypervolumes)
auto_mock.results_df.head()
auto_mock.hvs_df.head()

# To get the cluster in which each data point is in for a given individual, labels attribute
print(auto_mock.labels[0])

# Select the best solution in the population. If ARIs were computed, then use this metric, 
# otherwise select the closest solution to the origin (the default metric is the euclidean distance)
auto_mock.select_solution()
print(auto_mock.best_individual)

# Visualise the resulting pareto front
auto_mock.plot_pareto(run=1, title='AutoMOCK Pareto Front')
```

## Classes
**AutoMOCK**(k_user, num_runs=1, num_gens=100, num_indvs=100, domain_delta='real', init_delta=95,
                 min_delta=80, max_delta=None, flexible_limits=0.3, stair_limits=None, crossover='uniform',
                 crossover_prob=1, delta_precision=1, mut_method='original', delta_mutation='gauss', squash='false',
                 delta_mutation_probability=1.0, delta_mutation_variance=0.01, delta_as_perct=True,
                 delta_inverse=False, L=10, L_comp=None, strategy='base', validate=False,
                 random_state=None, save_history=False, verbose=True)

* **k_user**: Rough estimate of number of clusters present in the dataset.
* **num_runs**: Number of independent runs to run the algorithm.
* **num_gens**: Number of generations to go trough.
* **num_indvs**: Size of the population.
* **domain_delta**: Whether to use "real" or "sr" (square root) encoding for delta.
* **init_delta**: Lower bound for delta during the first phase.
* **min_delta**: Final lower bound for delta.
* **max_delta**: Upper bound for delta.
* **flexible_limits**: Percentage (or number of) generations after which start using the "min_delta" parameter as the lower bound for delta.
* **stair_limits**: Number of generations after which trigger a delta lower bound change. If used, it will override flexible_limits parameter.
* **crossover**: Type of crossover to perform. Currently only "uniform" and None are supported.
* **crossover_prob**: Probability of an offspring of undergoing crossover.
* **delta_precision**: Number of decimal points to use in delta if encoding is "real", otherwise ignored.
* **mut_method**: Genotype mutation method to use. Currently "original", "centroid" and "neighbour" are supported.
* **delta_mutation**: How to mutate delta. Currently "gauss", "uniform" and "random" are supported.
* **squash**: Whether to squash the delta mutation distribution to ensure an equal mass of probability.
* **delta_mutation_probability**: Probability of an offspring of having its delta mutated.
* **delta_mutation_variance**: Variance of the delta mutation probability distribution.
* **delta_as_perct**: Interpret delta_mutation_variance as a percentage of delta.
* **delta_inverse**: If the variance of delta is given as a percentage and True, then assign greater variance to low values of delta.
* **L**: Number of nearest neighbours to look at.
* **L_comp**: Unused.
* **strategy**: Unused.
* **validate**: Not yet implemented.
* **random_state**: Random seed.
* **save_history**: Whether to save results for every generation (True) or only the last one (False).
* **verbose**: Whether to report progress (True) or not (False)

**DMOCK**(k_user, num_runs=1, num_gens=100, num_indvs=100, delta=95, crossover='uniform',
                 crossover_prob=1.0, mut_method='original', L=10, L_comp=None, validate=False,
                 random_state=None, save_history=False, verbose=True)
                 
**MOCK**(k_user, num_runs=1, num_gens=100, num_indvs=100, crossover='uniform',
                 crossover_prob=1.0, mut_method='original', L=10, L_comp=None, validate=False,
                 random_state=None, save_history=False, verbose=True):