# TODO: Horrible bug which makes the pareto front look bad
# TODO: Maybe move multi_run to AutoMOCK class?
# TODO: Consolidate these classes with class Datapoint
# TODO: Eliminate the need for having mock_args
# TODO: Add validation
# TODO: Remove not used functions
# TODO: Clean duplicated code
# TODO: Optimise speed in DMOCK and MOCK

import numpy as np
import matplotlib.pyplot as plt
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from utils import check_config
from multi_run_mock import multi_run_mock


class AutoMOCK:
    def __init__(self, k_user, num_runs=1, num_gens=100, num_indvs=100, domain_delta='real', init_delta=95,
                 min_delta=80, max_delta=None, flexible_limits=0.3, stair_limits=None, crossover='uniform',
                 crossover_prob=1, delta_precision=1, mut_method='original', delta_mutation='gauss', squash='false',
                 delta_mutation_probability=1.0, delta_mutation_variance=0.01, delta_as_perct=True,
                 delta_inverse=False, L=10, L_comp=None, strategy='base', validate=False,
                 random_state=None, save_history=False, verbose=True):
        self.k_user = k_user
        self.num_runs = num_runs
        self.num_gens = num_gens
        self.num_indvs = num_indvs
        self.domain_delta = domain_delta
        self.init_delta = init_delta
        self.min_delta = min_delta
        self.max_delta = max_delta
        self.flexible_limits = flexible_limits
        self.stair_limits = stair_limits
        self.crossover = crossover
        self.crossover_prob = crossover_prob
        self.delta_precision = delta_precision
        self.mut_method = mut_method
        self.delta_mutation = delta_mutation
        self.squash = squash
        self.delta_mutation_probability = delta_mutation_probability
        self.delta_mutation_variance = delta_mutation_variance
        self.delta_as_perct = delta_as_perct
        self.delta_inverse = delta_inverse
        self.L = L
        self.L_comp = L_comp
        self.strategy = strategy
        self.validate = validate
        self.save_results = self.validate is False  # opposite of validate
        self.save_history = save_history
        self.save_labels = None
        self.verbose = verbose
        self.results_df = None
        self.hvs_df = None
        self.labels = None
        self.best_individual = None

        if random_state is None:
            self.random_state = np.random.randint(0, 9999999, size=num_runs)
        else:
            assert isinstance(random_state, (int, float, complex)) and not isinstance(random_state, bool),\
                "random_state must be numeric."
            self.random_state = [random_state*i for i in range(1, num_runs+1)]

        # Prepare list of runs
        self.runs_list = [{'seed': seed, 'run_number': run_number + 1}
                          for run_number, seed in enumerate(self.random_state)]

        # Variable used to standardize limit behaviour
        self.gens_step = 0.1

        # Check that the arguments are correct, or at least try to make them so.
        if not validate:
            check_config(self)

    def fit(self, X, y=None):
        multi_run_mock(self, X, y)

    def plot_pareto(self, run=1, title=None, show=True):
        self.check_fitted()

        mask = self.results_df.run == run
        plt.scatter(self.results_df[mask].VAR, self.results_df[mask].CNN)
        plt.xlabel('VAR')
        plt.ylabel('CNN')
        plt.title(title)

        if show:
            plt.show()

    def select_solution(self, ord=2):
        self.check_fitted()

        if np.isnan(self.results_df.ARI[0]):
            # Scale so one metric is not heavier than the other
            cnn = StandardScaler().fit_transform(self.results_df[['CNN']])
            var = StandardScaler().fit_transform(self.results_df[['VAR']])
            points = np.array([cnn, var])

            # Calculate the distances to the origin and select the closest solution
            dist = np.linalg.norm(points, ord=ord, axis=0)
            self.best_individual = self.results_df.iloc[np.argmin(dist)]
        else:
            # If there are ARIs, just select the individual with maximum ARI
            self.best_individual = self.results_df.iloc[self.results_df.ARI.idxmax()]

    def check_fitted(self):
        if self.results_df is None:
            raise NotFittedError


class DMOCK(AutoMOCK):
    def __init__(self, k_user, num_runs=1, num_gens=100, num_indvs=100, delta=95, crossover='uniform',
                 crossover_prob=1.0, mut_method='original', L=10, L_comp=None, validate=False,
                 random_state=None, save_history=False, verbose=True):
        super().__init__(k_user, num_runs=num_runs, num_gens=num_gens, num_indvs=num_indvs, crossover=crossover,
                         crossover_prob=crossover_prob, mut_method=mut_method, L=L, L_comp=L_comp,
                         validate=validate, random_state=random_state, save_history=save_history, verbose=verbose)
        self.init_delta = delta
        self.min_delta = delta
        self.max_delta = delta


class MOCK(AutoMOCK):
    def __init__(self, k_user, num_runs=1, num_gens=100, num_indvs=100, crossover='uniform',
                 crossover_prob=1.0, mut_method='original', L=10, L_comp=None, validate=False,
                 random_state=None, save_history=False, verbose=True):
        super().__init__(k_user, num_runs=num_runs, num_gens=num_gens, num_indvs=num_indvs, crossover=crossover,
                         crossover_prob=crossover_prob, mut_method=mut_method, L=L, L_comp=L_comp,
                         validate=validate, random_state=random_state, save_history=save_history, verbose=verbose)
        self.init_delta = 0
        self.min_delta = 0
        self.max_delta = 0
