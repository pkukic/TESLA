from dolfin import plot
import numpy as np
import pickle

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
            self.err = np.float64(1/constants.EPS_VAR)
        else:
            self.err = init_err
        self.goodness = -np.log10(self.err)

    @staticmethod    
    def eps():
        return abs(np.random.normal(0.0, constants.EPS_VAR))

    def evaluate(self):
        print(f"Evaluating: {self}")
        err = sim.error_from_n_sims(constants.N_ITER, self.heights_tuple)
        self.err = err + Unit.eps()
        self.goodness = -np.log10(self.err)
        return

    def crossover(self, other):
        assert isinstance(other, Unit)
        new_heights_list = []
        for (h1, h2) in zip(self.heights_tuple, other.heights_tuple):
            new_heights_list.append(round((h1 + h2) / 2))
        return Unit(tuple(new_heights_list), (self.err + other.err) / 2)

    def mutate(self, sgs):
        new_heights_arr = [0] * constants.N_POLY_TUNE
        for i, h in enumerate(self.heights_tuple):
            sgs_greater = sgs[i, :h]
            sgs_less = sgs[i, (h + 1):]
            sg_greater = np.sum(sgs_greater)
            sg_less = np.sum(sgs_less)
            p_greater = constants.P_MUT * sg_greater / (sg_greater + sg_less)
            p_less = constants.P_MUT * sg_less / (sg_greater + sg_less)
            p = np.random.uniform(0, 1)
            if p <= p_less:
                new_heights_arr[i] = round(sum(sgs_less[j] * (j + h + 1) 
                                               for j in range(len(sgs_less))) / sg_less)
            elif p > p_less and p <= p_less + p_greater:
                new_heights_arr[i] = round(sum(sgs_greater[j] * j 
                                               for j in range(len(sgs_greater))) / sg_greater)
        return Unit(tuple(new_heights_arr), self.err)

    def __repr__(self):
        return str(self.heights_tuple)
    

class Trainer:
    def __init__(self, iter):
        self.iter = iter
        self.sgs = np.zeros((constants.N_POLY_TUNE, constants.N_HEIGHTS))

        self.curr_i = 0
        self.from_pickle = False

        self.population = []
        self.best = None
        self.best_unit_error = np.float64(0)

    def initialize_population(self):
        for _ in range(constants.POPSIZE):
            u = Unit(tuple(np.random.randint(constants.N_HEIGHTS, size=constants.N_POLY_TUNE)))
            self.population.append(u)
        return
    
    def update_sgs(self, u):
        for i, h in enumerate(u.heights_tuple):
            self.sgs[i][h] += u.goodness
        return

    def selection(self):
        fitness_vals = np.array([u.goodness for u in self.population])
        order = fitness_vals.argsort()
        ranks = len(order) - order.argsort()
        ranks_exp = np.power(constants.C, ranks)
        ranks_sum = np.sum(ranks_exp)
        probabilities = ranks_exp / ranks_sum

        cumulative_probabilities = np.cumsum(probabilities)

        parent_indices = np.zeros((constants.POPSIZE - constants.ELITISM, 2), dtype=int)
        for i in range(constants.POPSIZE - constants.ELITISM):
            # Select two parents
            for j in range(2):
                rand_val = np.random.rand()
                parent_index = np.searchsorted(cumulative_probabilities, rand_val)
                parent_indices[i,j] = parent_index
        return parent_indices


    def train_step(self):
        # SINGLE PROCESS, SINGLE THREAD
        for i, u in enumerate(self.population):
            u.evaluate()
            self.update_sgs(u)
            print(f"Goodness of child {i} = {self.population[i]}: {u.goodness}")

        sorted_old_population = sorted(self.population, key=lambda u:u.goodness, reverse=True)            
        new_population = sorted_old_population[:constants.ELITISM]

        pairs = self.selection()
        for (i, j) in pairs:
            new_population.append(self.population[i]
                                  .crossover(self.population[j])
                                  .mutate(self.sgs))

        self.population = new_population[:]

        self.best = max(self.population, key=lambda u: float(u.goodness))
        self.best_unit_error = self.best.err
        return

    def save_population(self):
        pop_goodness = sum(u.goodness for u in self.population)
        pop_name = f"population_iter={self.curr_i}_pop_goodness={pop_goodness:.3f}.pickle"
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
    iters = 100
    t = Trainer(iters)
    t.train()

    b = t.best
    print(b)
    print(b.goodness)
    print(sum(u.goodness for u in t.population))

    m = domain.mesh(b.heights_tuple)

    plot(m)
    plt.gca().set_aspect('equal')
    plt.show()
