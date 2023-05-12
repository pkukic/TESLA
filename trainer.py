from dolfin import plot
import numpy as np

from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

import sim
import constants
import domain


class Unit:
    def __init__(self, heights_tuple, init_err = None):
        self.heights_tuple = heights_tuple
        if init_err is None:
            self.err = np.float64(1e10)
        else:
            self.err = init_err
        self.goodness = -np.log10(self.err)

    def evaluate(self):
        err = sim.error_from_n_sims(constants.N_SIMS_PER_UNIT, self.heights_tuple)
        self.err = err
        self.goodness = -np.log10(self.err)
        return

    def crossover(self, other):
        assert isinstance(other, Unit)
        new_heights_list = []
        for (h1, h2) in zip(self.heights_tuple, other.heights_tuple):
            new_heights_list.append(round((h1 + h2) / 2))
        return Unit(tuple(new_heights_list))
    
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
        return Unit(tuple(lower * new_heights_arr + higher * old_heights_arr))
    
    def __repr__(self):
        return str(self.heights_tuple)

def evaluate_unit(unit):
    unit.evaluate()
    return

class Trainer:
    def __init__(self, elitism, p, jump_dist, iter):
        self.popsize = constants.POPSIZE
        self.elitism = elitism
        self.p = p
        self.jump_dist = jump_dist
        self.iter = iter
        
        self.population = []
        self.best = None
        self.best_unit_error = np.float64(0)

    def initialize_population(self):
        for _ in range(self.popsize):
            u = Unit(tuple(np.random.randint(constants.N_HEIGHTS, size=constants.N_POLY_TUNE)))
            self.population.append(u)
        return
    
    def random_indices(self):
        n_children = self.popsize - self.elitism
        if n_children > 0:
            all_tuples = np.array([(i,j) for i in range(n_children) for j in range(i+1,n_children)])
            selected_tuples = np.random.choice(len(all_tuples), self.popsize, replace=False)
            return all_tuples[selected_tuples]
        return []
    
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
        for u in self.population:
            u.evaluate()
        self.population = sorted(self.population, key=lambda u: float(u.goodness), reverse=True)
        
        best_units = self.population[:self.elitism]
        new_population: best_units[:]
        
        # print(new_population)

        pairs = self.random_indices()
        for (i, j) in pairs:
            new_population.append(self.population[i].crossover(self.population[j]).mutate(self.p, self.jump_dist))
        
        new_population = sorted(new_population, key=lambda u:float(u.goodness), reverse=True)
        # print(new_population)

        self.population = new_population[:]

        self.best = max(self.population, key=lambda u: float(u.goodness))
        self.best_unit_error = self.best.err

        return

    def train(self):
        self.initialize_population()
        for i in range(self.iter):
            self.train_step()
            print(f"[Train error @{i + 1}]: {(self.best_unit_error):.6f} | Best unit: {self.best}\n")
        self.best = max(self.population, key=lambda u: float(u.goodness))
        return
    

if __name__ == '__main__':
    t = Trainer(0, 0.3, 2, 100)
    t.train()
    b = t.best
    m = domain.mesh(b.heights_tuple)

    plot(m)
    plt.gca().set_aspect('equal')
    plt.show()
