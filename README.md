# Structure

The `main*.py` and `artif*.py` files are the files that contain the primary EA code (the `main*.py` files is for the hypervolume-based adaptive strategy, and the artif*.py files are for the random- and interval-based adaptive strategies). Each of these groups of files then have a different seach method implemented, as a lot of local variables were required and the different search method modified different parts of the EA and so simple functions would not have sufficed.

The actual experiments are run through the generate*.py files, with different experimental configurations in each of the files (to ensure consistency and not having to remember every little detail of how the experiments differ and run the risk of not changing something for certain experiments - I did that once and lost more than a week of computation).

Most of the functions used within the EA are compartmentalised into files, namely: `precompute.py, initialisation.py, objectives.py, evaluation.py, operators.py, and classes.py`.
