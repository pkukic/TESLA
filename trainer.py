from dolfin import plot
import numpy as np
import pickle

from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

import sim
import constants
import domain

def eps():
    return abs(np.random.normal(0.0, 1e-9))

class Unit:
    def __init__(self, heights_tuple, init_err = None):
        self.heights_tuple = heights_tuple
        if init_err is None:
            self.err = np.float64(1e9)
        else:
            self.err = init_err
        self.goodness = -np.log10(self.err)

    def evaluate(self):
        print(f"Evaluating: {self}")
        err = sim.error_from_n_sims(constants.N_SIMS_PER_UNIT, self.heights_tuple)
        self.err = err + eps()
        self.goodness = -np.log10(self.err)
        return

    def crossover(self, other):
        assert isinstance(other, Unit)
        new_heights_list = []
        for (h1, h2) in zip(self.heights_tuple, other.heights_tuple):
            new_heights_list.append(round((h1 + h2) / 2))
        return Unit(tuple(new_heights_list), (self.err + other.err) / 2)

    @staticmethod
    def _pick_new_height(height, jump_dist):
        height_low = np.clip(height - jump_dist, 0, constants.N_HEIGHTS - 1)
        height_high = np.clip(height + jump_dist, 0, constants.N_HEIGHTS - 1)
        return np.random.randint(height_low, height_high + 1)

    def mutate(self, p, jump_dist):
        new_heights_arr = [Unit._pick_new_height(self.heights_tuple[i], jump_dist)
                           for i in range(constants.N_POLY_TUNE)]
        old_heights_arr = list(self.heights_tuple)
        p_arr = np.random.uniform(0, 1, constants.N_POLY_TUNE)
        lower = p_arr < p
        higher = p_arr >= p
        return Unit(tuple(lower * new_heights_arr + higher * old_heights_arr), self.err)

    def __repr__(self):
        return str(self.heights_tuple)
    

class Trainer:
    def __init__(self, p, jump_dist, iter):
        self.popsize = constants.POPSIZE
        self.p = p
        self.jump_dist = jump_dist
        self.iter = iter

        self.curr_i = 0
        self.from_pickle = False

        self.population = []
        self.best = None
        self.best_unit_error = np.float64(0)

    def initialize_population(self):
        for _ in range(self.popsize):
            u = Unit(tuple(np.random.randint(constants.N_HEIGHTS, size=constants.N_POLY_TUNE)))
            self.population.append(u)
        return

    import numpy as np

    def roulette_wheel_selection(self):
        fitness_vals = np.array([u.goodness for u in self.population])
        sum_fitness = np.sum(fitness_vals)
        probabilities = fitness_vals / sum_fitness
        cumulative_probabilities = np.cumsum(probabilities)

        parent_indices = np.zeros((self.popsize, 2), dtype=int)
        for i in range(self.popsize):
            # select two parents
            for j in range(2):
                rand_val = np.random.rand()
                parent_index = np.searchsorted(cumulative_probabilities, rand_val)
                parent_indices[i,j] = parent_index
        return parent_indices


    def train_step(self):
        # MULTIPROCESSING
        # num_processes = 8
        # with Pool(processes=num_processes) as pool:
        #     self.population = list(pool.map(evaluate_unit, self.population, chunksize=int(constants.POPSIZE / num_processes)))

        # MULTITHREADING
        # num_workers = 4
        # with ThreadPoolExecutor(max_workers=num_workers) as executor:
        #     self.population = list(executor.map(evaluate_unit, self.population))

        # SINGLE PROCESS, SINGLE THREAD
        for i, u in enumerate(self.population):
            u.evaluate()
            # m = domain.mesh(u.heights_tuple)
            # plot(m)
            # plt.gca().set_aspect('equal')
            # plt.show()
            print(f"Goodness of child {i} = {self.population[i]}: {u.goodness}")

        new_population = []
        pairs = self.roulette_wheel_selection()
        for (i, j) in pairs:
            new_population.append(self.population[i].crossover(self.population[j]).mutate(self.p, self.jump_dist))

        new_population = sorted(new_population, key=lambda u:float(u.goodness), reverse=True)
        # print(new_population)

        self.population = new_population[:]

        self.best = max(self.population, key=lambda u: float(u.goodness))
        self.best_unit_error = self.best.err

        return

    def save_population(self):
        pop_name = f"population_iter={self.curr_i}_p={self.p}_jump_dist={self.jump_dist}_pop_goodness={(sum(u.goodness for u in self.population)):.3f}.pickle"
        with open(pop_name, 'wb+') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        return

    def train(self):
        if not self.from_pickle:
            self.initialize_population()
        for i_ in range(self.iter):
            self.curr_i += 1
            self.train_step()
            print(f"[Train error @{self.curr_i}]: {(self.best_unit_error):.6f} | Best unit: {self.best}\n")
            self.save_population()
        self.best = max(self.population, key=lambda u: float(u.goodness))
        return

    @staticmethod
    def load(fname):
        with open(fname, 'rb') as f:
            trainer = pickle.load(f)
        trainer.from_pickle = True
        return trainer


if __name__ == '__main__':
    # p = 0.1
    # jump_distance = 2
    # iters = 20
    # t = Trainer(p, jump_distance, iters)

    t = Trainer.load('/home/patrik/Drive/Current/Završni/Neuromorphic computing/Code/novo/TESLA/population_iter=3_p=0.1_jump_dist=2_goodness=9.200.pickle')

    print(t.population)

    t.train()
    b = t.best
    m = domain.mesh(b.heights_tuple)

    plot(m)
    plt.gca().set_aspect('equal')
    plt.show()
