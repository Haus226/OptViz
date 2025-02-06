import numpy as np
from abc import ABC, abstractmethod

class MetaHeuristic(ABC):
    def __init__(self, func:callable, var_dict:dict, bounds:np.array, pop_size:int, ttl:int):
        self.func = func
        self.pop_size = pop_size
        self.ttl = ttl
        self.var_dict = {v:0 for v in var_dict.keys()}
        self.lb, self.ub = bounds[:, 0], bounds[:, 1]
        self.age = np.ones((self.pop_size, )) * self.ttl if self.ttl > 0 else np.ones((self.pop_size, )) * np.inf
        self.dimensions = len(bounds)

        self.pop = np.random.uniform(self.lb, self.ub, (self.pop_size, len(bounds)))
        self.fitness = np.apply_along_axis(self.evaluate, 1, self.pop)

        best_idx = np.argmin(self.fitness)
        # Use copy, since they are referencing the same underlying array which may change
        self.optimal_coords = self.pop[best_idx].copy()
        self.optimal_evaluated = self.fitness[best_idx].copy()

    def evaluate(self, ind):
        for i, v in zip(ind, self.var_dict.keys()):
            self.var_dict[v] = i
        return self.func(**self.var_dict)
    
    @abstractmethod
    def step(self):
        pass

class DE(MetaHeuristic):
    def __init__(self, func, var_dict, bounds, pop_size, ttl, mut_1=0.9, mut_2=0.9, cross_p=0.95):
        super().__init__(func, var_dict, bounds, pop_size, ttl)
        self.mut_1 = mut_1
        self.mut_2 = mut_2
        self.cross_p = cross_p

    def step(self):
        for idx in range(self.pop_size):
            self.age[idx] -= 1
            idxs = [kdx for kdx in range(self.pop_size) if kdx != idx]
            a, b, c = self.pop[np.random.choice(idxs, 3, replace = False)].copy()
            mutant = a + self.mut_1 * (b - c) + self.mut_2 * (self.optimal_coords - a)
            mutant = np.clip(mutant, self.lb, self.ub)

            cross_points = np.random.rand(self.dimensions) < self.cross_p
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dimensions)] = True
            trial = np.where(cross_points, mutant, self.pop[idx])

            f = self.evaluate(trial)
            if f < self.fitness[idx] or (self.age[idx] == -1):
                self.fitness[idx] = f
                self.pop[idx] = trial
                self.age[idx] = self.ttl
                if f < self.optimal_evaluated:
                    self.optimal_coords = trial
                    self.optimal_evaluated = f

class HS(MetaHeuristic):
    def __init__(self, func, var_dict, bounds, pop_size, ttl, HMCR=0.7, PAR=0.3, BW=None):
        '''
        Harmony Search (HS) algorithm parameters:
        HMCR: Harmony Memory Considering Rate
        PAR: Pitch Adjusting Rate
        BW: Bandwidth
        '''
        super().__init__(func, var_dict, bounds, pop_size, ttl)
        self.HMCR = HMCR
        self.PAR = PAR
        self.BW = BW if BW is not None else self.ub - self.lb
        if self.BW is not None:
            assert(self.BW.shape[0] == self.dimensions)
        self.worst_idx = np.argmax(self.fitness)

    def step(self):
        r1_ = np.random.rand(self.pop_size, self.dimensions)
        r2_ = np.random.rand(self.pop_size, self.dimensions)
        r3_ = np.random.uniform(low=-1, high=1.001, size=(self.pop_size, self.dimensions))

        for idx in range(self.pop_size):
            self.age[idx] -= 1
            
            # Generate new harmony
            trial = np.zeros(self.dimensions)
            for kdx in range(self.dimensions):
                if r1_[idx][kdx] < self.HMCR:
                    # Memory consideration
                    trial[kdx] = self.pop[np.random.randint(0, self.pop_size)][kdx]
                else:
                    # Random consideration
                    trial[kdx] = np.random.uniform(self.lb[kdx], self.ub[kdx])
                
                # Pitch adjustment
                if r2_[idx][kdx] < self.PAR:
                    trial[kdx] += r3_[idx][kdx] * self.BW[kdx]
            
            trial = np.clip(trial, self.lb, self.ub)
            f = self.evaluate(trial)

            if (self.age[idx] == -1) or f < self.fitness[self.worst_idx]:
                self.pop[self.worst_idx] = trial
                self.fitness[self.worst_idx] = f
                self.age[self.worst_idx] = self.ttl
                if f < self.optimal_evaluated:
                    self.optimal_coords = trial.copy()
                    self.optimal_evaluated = f
                self.worst_idx = np.argmax(self.fitness)


class PSO(MetaHeuristic):
    def __init__(self, func, var_dict, bounds, pop_size, ttl, inertia=0.5, cognitive=1.5, social=1.5):
        super().__init__(func, var_dict, bounds, pop_size, ttl)
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        diff = np.fabs(self.lb - self.ub)
        self.velocity = np.random.uniform(-diff, diff, (self.pop_size, self.dimensions))
        self.personal_best = self.pop.copy()

    def step(self):
        self.age -= 1
        r_p = np.random.rand(self.pop_size, self.dimensions)
        r_s = np.random.rand(self.pop_size, self.dimensions)

        self.velocity = (
            self.inertia * self.velocity + 
            self.cognitive * r_p * (self.personal_best - self.pop) + 
            self.social * r_s * (self.optimal_coords - self.pop)
        )
        self.pop += self.velocity
        self.pop = np.clip(self.pop, self.lb, self.ub)

        fitness = np.apply_along_axis(self.evaluate, 1, self.pop)
        update_mask = (fitness < self.fitness) | (self.age == -1)
        self.personal_best[update_mask] = self.pop[update_mask]
        self.age[update_mask] = self.ttl
        self.fitness = fitness
        
        # Find and update global optimum
        best_idx = np.argmin(self.fitness)
        if self.fitness[best_idx] < self.optimal_evaluated:
            self.optimal_coords = self.personal_best[best_idx].copy()
            self.optimal_evaluated = self.fitness[best_idx].copy()

class ArtBC(MetaHeuristic):
    def __init__(self, func, var_dict, bounds, pop_size, ttl):
        super().__init__(func, var_dict, bounds, pop_size, ttl)

    def bees(self, bee_type="employed"):
        r = range(self.pop_size // 2) if bee_type == "employed" else range(self.pop_size // 2, self.pop_size)
        probabilities = np.exp(-self.fitness) / np.sum(np.exp(-self.fitness)) if bee_type == "onlook" else None
        for idx in r:
            mask = [i for i in range(self.pop_size) if i != idx]
            self.age[idx] -= 1
            jdx = np.random.choice(mask, p=probabilities[mask] if bee_type == "onlook" else None)
            kdx = np.random.randint(0, self.dimensions)

            trial = self.pop[idx].copy()
            phi = np.random.uniform(-1, 1)
            trial[kdx] = self.pop[idx][kdx] + phi * (self.pop[idx][kdx] - self.pop[jdx][kdx])
            trial = np.clip(trial, self.lb, self.ub)
            
            f = self.evaluate(trial)
            if f < self.fitness[idx]:
                self.pop[idx] = trial
                self.fitness[idx] = f
                self.age[idx] = self.ttl
                if f < self.optimal_evaluated:
                    self.optimal_coords = trial.copy()
                    self.optimal_evaluated = f
            if self.age[idx] == -1:
                self.pop[idx] = self.lb + np.random.uniform(0, 1) * (self.ub - self.lb)
                self.fitness[idx] = self.evaluate(self.pop[idx])
                self.age[idx] = self.ttl
                if self.fitness[idx] < self.optimal_evaluated:
                    self.optimal_coords = self.pop[idx].copy()
                    self.optimal_evaluated = self.fitness[idx]

    def step(self):
        self.bees()
        self.bees("onlook")
        
