import numpy as np
import pandas as pd
import copy
from time import time
from sklearn.model_selection import cross_val_score
from sklearn import svm
import random
class Chromosome:
    def __init__(self, FeatureSubset_En):
        # Switcher: cross-validate, group-filter,...
        self.FeatureSubset_En = FeatureSubset_En
        self.decode()
        ScoreSwitcher = {
            'accuracy_score': self.accuracy_score,
            'auc_score': self.auc_score,
            'f1_score': self.f1_score,
            'recall_score': self.recall_score,
            'precision_score': self.precision_score,
            'group_score': self.group_score,
        }

        self.Score = ScoreSwitcher.get(
                    self.ConfigChromosome['ScoreType'],
                    lambda: 'Invalid score type')()

    def encode(self):
        pass

    def decode(self):
        self.FeatureSubset = []
        for (i, j) in zip(self.FeatureSubset_En, self.FullFeatures):
            if i == 1:
                self.FeatureSubset.append(j)

    def accuracy_score(self):
        X = self.Table[self.FeatureSubset]
        y = self.Table[self.TargetFeature]
        model = svm.SVC(gamma='scale')
        accuracy = cross_val_score(model, X.values, y.values.reshape(-1), scoring='accuracy', cv = 10)
        self._fitness = accuracy.mean()

    def group_score(self):
        pass

    def auc_score(self):
        pass

    def f1_score(self):
        pass

    def recall_score(self):
        pass

    def precision_score(self):
        pass

    @classmethod
    def ConfigInit(cls, ConfigChromosome):
        cls.ConfigChromosome = ConfigChromosome
        cls.FullFeatures = cls.ConfigChromosome['FullFeatures']
        cls.Table = cls.ConfigChromosome['Table']
        cls.TargetFeature = cls.ConfigChromosome['TargetFeature']

class Population:
    def __init__(self, first_population: list = None):
        if first_population is None:
            first_population = []
        self._population_size = len(first_population)
        self._generations = [first_population]
        _fitness = [chromo._fitness for chromo in first_population]
        self._generations_solution = [first_population[np.argmax(_fitness)]]
        self.best_fitness = np.max(np.asarray(_fitness))
        self._all_best_fitness = [self.best_fitness]
        self.best_solution = self._generations_solution[-1]
        self.verbose = False

    def mutation(self, children):
        new_children = []
        chosen_indexes = np.random.choice(len(children), size=self._mutation_size, replace=False)
        for i in range(len(children)):
            if i not in chosen_indexes:
                new_children.append(copy.deepcopy(children[i]))
                continue
            chromosome = children[i]
            if self.verbose > 1:
                print('\t\tStarting mutation {}th child'.format(len(new_children) + 1))
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
            if self.verbose > 1:
                print('\t\t{}_th 2 childs'.format(i + 1))
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
        fitness = np.asarray([chromo._fitness for chromo in generation])
        fitness /= np.sum(fitness)
        parents = []
        for _ in range(self._offspring_number):
            chosen_indexes = np.random.choice(np.arange(len(fitness)), size=k, replace=False, p=fitness)
            best_index = chosen_indexes[np.argmax(fitness[chosen_indexes])]
            parents.append(copy.deepcopy(generation[best_index]))
        return parents

    def tournament_selection(self, generation, k):
        fitness = np.asarray([chromo._fitness for chromo in generation])
        parents = []
        for _ in range(self._offspring_number):
            chosen_indexes = np.random.choice(self._population_size, size=k, replace=False)
            best_index = chosen_indexes[np.argmax(fitness[chosen_indexes])]
            parents.append(copy.deepcopy(generation[best_index]))
        return parents

    def generate_next_population(self, config: dict, verbose=False):

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
        self.verbose = verbose


        if self.verbose > 0:
            print('\nIteration {}'.format(len(self._all_best_fitness)))
            # print('\nIteration {} - Record {}'.format(len(self._all_best_fitness), self._best_solution._fitness))
        generation = self._generations[-1]
        np.random.shuffle(generation)
        generation_fitness = np.asarray([chromo._fitness for chromo in generation])

        # selection phase
        selection_switcher = {
            'roulette_wheel': self.roulette_wheel_selection,
            'tournament': self.tournament_selection
        }
        parents = selection_switcher.get(
                    self._selection_method['type'],
                    lambda: 'Invalid selection method')(
                        generation,
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

        #
        new_generation = generation + new_children
        new_generation_fitness = np.asarray([chromo._fitness for chromo in new_generation])

        sorted_indexes = np.argsort(new_generation_fitness)
        best_indexes = sorted_indexes[-self._population_size:]
        best_indexes.sort()

        for idx in range(len(new_generation)):
            if idx not in best_indexes:
                np.delete(new_generation, idx)
                np.delete(new_generation_fitness, idx)
        self._generations.append(new_generation)
        self._all_best_fitness.append(np.max(new_generation_fitness))
        self._generations_solution.append(new_generation[np.argmax(new_generation_fitness)])
        new_best_fitness = self._all_best_fitness[-1]

        if new_best_fitness > self.best_fitness:
            self.best_solution = self._generations_solution[-1]
            self.best_fitness = self._all_best_fitness[-1]

    @classmethod
    def population_initialization(cls, config_init, population_size=100, genes_number: int = None):
        Chromosome.ConfigInit(config_init)

        new_population = np.random.randint(2, size = (population_size, len(config_init['FullFeatures'])))

        return cls([Chromosome(chromo) for chromo in new_population])

class MPGA:
    def __init__(self, first_GAs: list = None):
        if first_GAs is None:
            first_GAs = []
        self.GAs = first_GAs # [GA, GA, ..., GA]
        self.verbose = False
        self._size_interact = 30 # Chosen random number: chromos GA interact chromos other GA
        self._best_fitness = max([i.best_fitness for i in self.GAs])
        self.all_best_fitness = [self._best_fitness]
        self._best_solution = [i.best_solution for i in self.GAs][np.argmax([i.best_fitness for i in self.GAs])]

    def generate_next_gas(self, config):
        for ga in [i for i in self.GAs]:
            ga.generate_next_population(config) # GA ----> Selection, Cross, Mutation ---> Update GA._generations
        # interact
        for ga, id in zip([i._generations[-1] for i in self.GAs], range(len(self.GAs))):
            # Chosen index GAs without ga:
            chosen_id = random.choice([num for num in range(len(self.GAs)) if num != id])
            chosen_ga = self.GAs[chosen_id]._generations[-1]
            new_ga = self.interact_gas(ga, chosen_ga)
            fitness_new_ga = [i._fitness for i in new_ga]
            best_fitness_new_ga = np.max(np.asarray(fitness_new_ga))
            self.GAs[id]._generations.append(new_ga)
            self.GAs[id]._all_best_fitness.append(best_fitness_new_ga)
            if best_fitness_new_ga > self.GAs[id].best_fitness:
                self.GAs[id].best_fitness = best_fitness_new_ga
                self.GAs[id].best_solution = [new_ga[np.argmax(best_fitness_new_ga)]]

    def generate_gas(self, config):
        self.generations_number = config['generations_number']
        for epoch in range(self.generations_number):
            print('Iteration: ', epoch)
            self.generate_next_gas(config)
            best_fitness = max([i.best_fitness for i in self.GAs])

            self.all_best_fitness.append(best_fitness)

            if best_fitness > self._best_fitness:
                print('improved : ', best_fitness)
                self._best_fitness = best_fitness
                self._best_solution = [i.best_solution for i in self.GAs][np.argmax([i.best_fitness for i in self.GAs])]
            else:
                print('not improved: ', self._best_fitness)

        return self._best_solution, self._best_fitness, self.all_best_fitness

    def interact_gas(self, ga, chosen_ga):

        fitness_chosen_ga = np.asarray([chromo._fitness for chromo in chosen_ga])
        fitness_chosen_ga /= np.sum(fitness_chosen_ga)
        best_chromosomes_indexes = np.random.choice(np.arange(len(fitness_chosen_ga)), size=self._size_interact, replace=False, p=fitness_chosen_ga)
        new_ga = ga
        for id in best_chromosomes_indexes:
            new_ga.append(chosen_ga[id])
        new_ga_fitness = np.asarray([chromo._fitness for chromo in new_ga])

        sorted_indexes = np.argsort(new_ga_fitness)
        best_indexes = sorted_indexes[-len(chosen_ga):]
        best_indexes.sort()

        for idx in range(len(new_ga)):
            if idx not in best_indexes:
                np.delete(new_ga, idx)
                np.delete(new_ga_fitness, idx)
        return new_ga

    @classmethod
    def gas_initialization(cls, config_init, population_size, gas_size=10):
        first_GAs = []
        for i in range(gas_size):
            population = Population.population_initialization(config_init, population_size)
            print(type(population))
            first_GAs.append(population)
        return cls(first_GAs)
