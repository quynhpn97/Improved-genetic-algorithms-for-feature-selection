# GA-Initial
import numpy as np
import pandas as pd
import copy
from time import time
from sklearn.model_selection import cross_val_score
from sklearn import svm
import random

class GaInitialization:
    def __init__(self, config_cross, config_mutation):
        self.config_cross = config_cross
        self.config_mutation = config_mutation

    def roulette_wheel_selection(self, generation, offspring_number):
        parents = []
        for _ in range(offspring_number):
            chosen_index = np.random.choice(np.arange(len(generation)), size=1, replace=False, p=None)[0]
            parents.append(copy.deepcopy(generation[chosen_index]))
        return parents

    def mutation(self, children):
        new_children = []
        chosen_indexes = np.random.choice(len(children), size=self.config_mutation['_mutation_size'], replace=False)
        for i in range(len(children)):
            if i not in chosen_indexes:
                new_children.append(children[i])
            else:
                chromosome = children[i]
                _mutation = bool(np.random.rand(1) <= self.config_mutation['_mutation_probability'])
                if _mutation:
                    mutation_genes_indexe = np.random.choice(
                                                np.arange(len(chromosome.FeatureSubset_En)),
                                                size=1,
                                                replace=False)
                    new_FeatureSubset_En = np.array(chromosome.FeatureSubset_En)
                    new_FeatureSubset_En[mutation_genes_indexe] = 1-chromosome.FeatureSubset_En[mutation_genes_indexe]
                    new_children.append(Chromosome(new_FeatureSubset_En))
                else:
                    new_children.append(children[i])
        return new_children

    def crossover_1point(self, generation):
        parents = self.roulette_wheel_selection(generation, self.config_cross['_offspring_number'])
        children = []
        for i in range(int(self.config_cross['_offspring_number'] / 2)):
            if True > 1:
                print('\t\t{}_th 2 childs'.format(i + 1))
            _crossover = bool(np.random.rand(1) <= self.config_cross['_crossover_probability'])
            if _crossover:
                index = np.random.randint(len(parents))
                father = parents.pop(index)
                index = np.random.randint(len(parents))
                mother = parents.pop(index)
                if self.config_cross['_cross_position'] == None:
                    config_position = np.random.randint(len(mother.FeatureSubset_En))
                child_1 = Chromosome(np.array(list(father.FeatureSubset_En[:config_position])+list(mother.FeatureSubset_En[config_position:])))
                child_2 = Chromosome(np.array(list(mother.FeatureSubset_En[:config_position])+list(father.FeatureSubset_En[config_position:])))
                children.append(child_1)
                children.append(child_2)
        return children

    def initial(self, config_init, population_size):
        first_population = {'P':[], 'Q': []}
        Chromosome.ConfigInit(config_init)
        new_population = np.random.randint(2, size = (population_size, len(config_init['FullFeatures'])))

        P0 = [Chromosome(chromo) for chromo in new_population]
        first_population['P'].append(P0)

        # Trao đổi chéo + Đột biến
        children = self.crossover_1point(P0)

        new_children = self.mutation(children)

        # Q
        first_population['Q'].append(new_children)

        return first_population


import random
# check DominatedSolution
def Dominated(x, y):
    u = x._fitness
    v = y._fitness
    flag = 0
    for i, j in zip(u, v):
      if i >= j:
        flag += 1
    if flag == len(u):
      for i, j in zip(u, v):
        if i > j:
          return 1
    if flag < len(u):
      return 0

class Chromosome:
    def __init__(self, FeatureSubset_En):
        # Switcher: cross-validate, group-filter,...
        self.model = svm.SVC(gamma='scale')
        self.cd_fitness = 0
        self.FeatureSubset_En = FeatureSubset_En
        self.decode()
        self._fitness = []
        ScoreSwitcher = {
            'accuracy_score': self.accuracy_score,
            'number_feature_subset': self.number_feature_subset,
        }

        for i in self.ConfigChromosome['ScoreType']:
            self._fitness.append(ScoreSwitcher.get(
                    i,
                    lambda: 'Invalid score type')())

    def encode(self):
        pass

    def decode(self):
        self.FeatureSubset = []
        for i, j in zip(self.FeatureSubset_En, self.FullFeatures):
            if i == 1:
                self.FeatureSubset.append(j)

    def accuracy_score(self):
        '''
        X = self.Table[self.FeatureSubset]
        y = self.Table[self.TargetFeature]
        accuracy = cross_val_score(self.model, X.values, y.values.reshape(-1), scoring='roc_auc', cv = 10)
        return accuracy.mean()
        '''
        return len(self.FeatureSubset_En) - len(self.FeatureSubset)

    def number_feature_subset(self):
        X = self.Table[self.FeatureSubset]
        y = self.Table[self.TargetFeature]
        model = svm.SVC(gamma='scale')
        accuracy = cross_val_score(self.model, X.values, y.values.reshape(-1), scoring='f1', cv = 10)
        return accuracy.mean()

    @classmethod
    def ConfigInit(cls, ConfigChromosome):
        cls.ConfigChromosome = ConfigChromosome
        cls.FullFeatures = cls.ConfigChromosome['FullFeatures']
        cls.Table = cls.ConfigChromosome['Table']
        cls.TargetFeature = cls.ConfigChromosome['TargetFeature']

class MOGA:

    def __init__(self, first_population: dict = None):
        if first_population is None:
            first_population = {'P': [], 'Q': []}
        self._generations = first_population
        self._population_size = 100

    def IdentifyParatoFront(self, population):
        F = {}
        F_id = {}
        i = 1
        F['F'+str(i)] = []
        F_id['F'+str(i)] = []
        n = [0] * len(population)
        S = {}
        S_id = {}
        for p, id_p in zip(population, range(len(population))):
            S[str(id_p)] = []
            S_id[str(id_p)] = []
            n[id_p] = 0
            for q, id_q in zip(population, range(len(population))):
                if Dominated(p, q):
                    S[str(id_p)].append(q)
                    S_id[str(id_p)].append(id_q)
                elif Dominated(q, p):
                    n[id_p] += 1
            if n[id_p] == 0:
                F['F'+str(i)].append(p)
                F_id['F'+str(i)].append(id_p)

        while len(F['F'+str(i)]) !=0 :
            Q = []
            Q_id = []
            for p, id_p in zip(F['F'+str(i)], F_id['F'+str(i)]):
                for q, id_q in zip(S[str(id_p)], S_id[str(id_p)]):
                    n[id_q] -= 1
                    if n[id_q] == 0:
                        Q.append(q)
                        Q_id.append(id_q)
            i += 1
            F['F'+str(i)] = Q
            F_id['F'+str(i)] = Q_id
        del F['F'+str(i)]
        return F

    def CrowdingDistance(self, F):
        for i in range(1, len(F)+1):
            F_i = F['F'+str(i)]
            if len(F_i)>2:
                fitness = np.array([chrome._fitness for chrome in F_i])
                cd_i = np.zeros(fitness.shape)
                cd_i = cd_i+0.1
                for k in range(fitness.shape[1]):
                    if len(np.unique(fitness[:, k])) != 1:
                        li = fitness[:,k]
                        min_value = min(li)
                        mi_idxes = [i for i, x in enumerate(li) if x == min_value]
                        max_value = max(li)
                        ma_idxes = [i for i, x in enumerate(li) if x == max_value]
                        for id in range(len(li)):
                            if (id not in mi_idxes):
                                if id not in ma_idxes:
                                    cd_i[id,k] = (min(filter(lambda x: x > fitness[id, k],list(fitness[:, k]))) - min(filter(lambda x: x < fitness[id, k],fitness[:, k])))/(max(fitness[:, k])-min(fitness[:, k]))
                cd_fitness = np.sum(cd_i, axis = 1)
                for chrome, i in zip(F_i, range(cd_fitness.shape[0])):
                    chrome.cd_fitness = cd_fitness[i]

    def generate_next_population(self):
        self.R = self._generations['P'][-1] + self._generations['Q'][-1]
        # Identify Parato Front F1, F2, ..., FK
        F = self.IdentifyParatoFront(self.R)

        next_population = []
        for k in range(1, len(F)+1):
            if len(next_population)+len(F['F'+str(k)]) <= self._population_size:
                next_population.extend(F['F'+str(k)])

        self._generations['P'].append(next_population)
        # Crowding Distance F1, F2, ..., Fk and cd_fitness
        self.CrowdingDistance(F)

        # selection phase
        selection_switcher = {
            'roulette_wheel': self.roulette_wheel_selection,
            'tournament': self.tournament_selection
        }
        parents = selection_switcher.get(
                    self._selection_method['type'],
                    lambda: 'Invalid selection method')(
                        next_population,
                        self._selection_method['k']
                    )

        # cross-over phase
        crossover_switcher = {
            '1point': self.crossover_1point,
        }
        children = crossover_switcher.get(
                    self._crossover_method['type'],
                    lambda: 'Invalid crossover method')(
                        parents,
                        self._crossover_method['parameters']
                    )
        # mutation phase
        new_children = self.mutation(children)

        # Update Q
        self._generations['Q'].append(new_children)

    def mutation(self, children):
        new_children = []
        chosen_indexes = np.random.choice(len(children), size=self._mutation_size, replace=False)
        for i in range(len(children)):
            if i not in chosen_indexes:
                new_children.append(copy.deepcopy(children[i]))
                continue
            chromosome = children[i]
            _mutation = bool(np.random.rand(1) <= self._mutation_probability)
            if _mutation:
                mutation_genes_indexe = np.random.choice(
                                            np.arange(len(chromosome.FeatureSubset_En)),
                                            size=1,
                                            replace=False)
                new_FeatureSubset_En = np.array(chromosome.FeatureSubset_En)
                new_FeatureSubset_En[mutation_genes_indexe] = 1-chromosome.FeatureSubset_En[mutation_genes_indexe]
                new_children.append(Chromosome(new_FeatureSubset_En))
        return new_children

    def crossover_1point(self, parents, cross_position):
        children = []
        for i in range(int(self._offspring_number / 2)):
            _crossover = bool(np.random.rand(1) <= self._crossover_probability)
            if _crossover:
                index = np.random.randint(len(parents))
                father = parents.pop(index)
                index = np.random.randint(len(parents))
                mother = parents.pop(index)
                if cross_position == None:
                    cross_position = np.random.randint(len(mother.FeatureSubset_En))
                child_1 = Chromosome(np.array(list(father.FeatureSubset_En[:cross_position])+list(mother.FeatureSubset_En[cross_position:])))
                child_2 = Chromosome(np.array(list(mother.FeatureSubset_En[:cross_position])+list(father.FeatureSubset_En[cross_position:])))
                children.append(child_1)
                children.append(child_2)
        return children

    def roulette_wheel_selection(self, generation, k=5):
        fitness = np.asarray([chromo.cd_fitness for chromo in generation])
        fitness /= np.sum(fitness)
        parents = []
        for _ in range(self._offspring_number):
            chosen_indexes = np.random.choice(np.arange(len(fitness)), size=k, replace=False, p=fitness)
            best_index = chosen_indexes[np.argmax(fitness[chosen_indexes])]
            parents.append(copy.deepcopy(generation[best_index]))
        return parents

    def tournament_selection(self, generation, k):
        fitness = np.asarray([chromo.cd_fitness for chromo in generation])
        parents = []
        for _ in range(self._offspring_number):
            chosen_indexes = np.random.choice(self._population_size, size=k, replace=False)
            best_index = chosen_indexes[np.argmax(fitness[chosen_indexes])]
            parents.append(copy.deepcopy(generation[best_index]))
        return parents

    def generate_populations(self, config: dict, verbose=False):
        self._offspring_number = int(self._population_size * config['offspring_ratio'])
        if self._offspring_number % 2 == 1:
            self._offspring_number += 1
        self._crossover_probability = config['crossover_probability']
        self._selection_method = config['selection_method']
        self._crossover_method = config['crossover_method']
        self._mutation_probability = config['mutation_probability']
        self._mutation_size = int(self._offspring_number * config['mutation_ratio'])
        self._chromosomes_replace = self._offspring_number
        self._generations_number = config['generations_number']
        self._stop_criterion_depth = config['stop_criterion_depth']

        depth = 0
        for epoch in range(self._generations_number):
            self.generate_next_population()
        return [list(i.FeatureSubset_En) for i in self._generations['P'][-1]]

    @classmethod
    def population_initialization(cls, config_init, population_size):
        config_cross = {'_cross_position': None, '_crossover_probability': 0.9, '_offspring_number': 50}
        config_mutation = {'_mutation_size': 10, '_mutation_probability': 0.1}
        ga_init = GaInitialization(config_cross, config_mutation)
        first_population = ga_init.initial(config_init, population_size)
        return cls(first_population)
