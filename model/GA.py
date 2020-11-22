import numpy as np
import pandas as pd
import copy
from time import time
from sklearn.model_selection import cross_val_score
from sklearn import svm
#from Filter-Approach import group-filter as GF
#from Filter-Approach import ranking_filter as RF

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
        self._best_fitness = np.max(np.asarray(_fitness))
        self._all_best_fitness = [self._best_fitness]
        self._generations_solution = [first_population[np.argmax(_fitness)]]
        self._best_solution = self._generations_solution[-1]
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

    def generate_next_population(self, config):

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
        #self.verbose = verbose

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
        if self.verbose > 1:
            print('----CROSS-OVER PHASE')
            start_time = time()
        crossover_switcher = {
            '1point': self.crossover_1point,
        }
        children = crossover_switcher.get(
                    self._crossover_method['type'],
                    lambda: 'Invalid crossover method')(
                        parents,
                        self._crossover_method['parameters']
                    )
        best_fitness = np.max([chromo._fitness for chromo in children])
        if self.verbose > 0:
            if self.verbose > 1:
                print('Time of cross-over: {} seconds'.format(time() - start_time))
            print('\tCROSS-OVER best fitness: {}'.format(best_fitness))

        # mutation phase
        if self.verbose > 1:
            print('****MUTATION PHASE')
            start_time = time()
        new_children = self.mutation(children)
        best_fitness = np.max(np.asarray([chromo._fitness for chromo in new_children]))
        if self.verbose > 0:
            if self.verbose > 1:
                print('Time of mutation: {} seconds'.format(time() - start_time))
            print('\tMUTATION best fitness: {}'.format(best_fitness))

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
        # if self._all_best_fitness[-1] < self._best_fitness:
        #     self._best_solution = self._generations_solution[-1]
        #     self._best_fitness = self._all_best_fitness[-1]
        return self._all_best_fitness[-1]

    def print(self):
        print('Population size: ' + str(self._population_size))
        print('Offspring number: ' + str(self._offspring_number))
        print('Selection type: ' + self._selection_method['type'].capitalize())
        print('Crossover method: ' + self._crossover_method['type'].capitalize())
        print('Crossover probability: ' + str(self._crossover_probability))
        print('Mutation probability: ' + str(self._mutation_probability))
        print('Mutation size: ' + str(self._mutation_size))
        print('Max generations number: ' + str(self._generations_number))
        print('Stop criterion depth: ' + str(self._stop_criterion_depth), end='\n\n')


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
        self.verbose = verbose
        self.print()

        print('Initial fitness: {}'.format(self._best_fitness))
        depth = 0
        for epoch in range(self._generations_number):
            new_best_fitness = self.generate_next_population(config)
            print('\tGeneration {}: fitness {}'.format(epoch + 1, new_best_fitness))
            if new_best_fitness > self._best_fitness:
                if self.verbose > 0:
                    self._best_fitness = new_best_fitness
                    self._best_solution = self._generations_solution[-1]
                    print('\tFitness improved for {} generations'.format(depth))
                    depth = 0
                depth += 1
                if depth > self._stop_criterion_depth:
                    if self.verbose > 0:
                        print('**********STOP CRITERION DEPTH REACHED**********')
                    break
            else:
                depth += 1
                if self.verbose > 0:
                    print('\tFitness not improved for {} generations'.format(depth))
                if depth > self._stop_criterion_depth:
                    if self.verbose > 0:
                        print('**********STOP CRITERION DEPTH REACHED**********')
                    break
        return self._best_solution, self._best_fitness, self._all_best_fitness

    @classmethod
    def population_initialization(cls, config_init, new_population, population_size=100, genes_number: int = None):
        Chromosome.ConfigInit(config_init)
        if new_population == None:
            new_population = np.random.randint(2, size = (population_size, len(config_init['FullFeatures'])))
        return cls([Chromosome(chromo) for chromo in new_population])
