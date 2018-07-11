# Delta-MOCK in Python
This is Python code for Delta-MOCK, including Adaptive Delta-MOCK, used in the paper *"Towards an Adaptive Encoding for Evolutionary Data Clustering"*<sup>1</sup>.

<sup>1</sup>Cameron Shand, Richard Allmendinger, Julia Handl, and John Keane. 2018. Towards an Adaptive Encoding for Evolutionary Data Clustering. In GECCO’18: Genetic and Evolutionary Computation Conference, July 15–19, 2018, Kyoto, Japan. ACM, New York, NY, USA, 8 pages. https://doi.org/10.1145/3205455.3205506

## Requirements
A `setup.py` is on the to-do list to enforce requirements, but some key requirements will be noted here:
* Python 3.6+ 
* DEAP 1.2
* python-igraph (best way is to install through conda-forge)
* Usual python stack (numpy, scipy etc.)

## File Explanation
I will try here to explain the basic premise and flow of running MOCK. The single python file that is run is the `run_mock.py` file, from which everything is executed. Some of the main parameters for MOCK can be adjusted in the `mock_config.json` file. If you would like more customisation/interaction through this config file, let me know.

The `run_mock()` function is the centrepoint. From here, the data is loaded, processed, and everything is set up to run MOCK. It has options to validate (compare to previous results to ensure consistent behaviour of MOCK) and save results.

The main logic of MOCK itself can be found in the `delta_mock.py` file, where we create the DEAP toolbox (and register all the functions, such as mutation, initialisation etc.) and actually run the EA. Most of the files should be self-explanatory. The mutation and crossover operators can be found in `operators.py`, the objectives are in `objectives.py`, the initialisation is in `initialisation.py`. 

Some files, such as `graph_funcs.py` and `analyse_funcs.py` are rather specific, and not particularly useful to other people/work/data. On the to-do list is to clean up these files and abstract them out a little more.