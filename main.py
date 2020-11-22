from model.GA import Population
from model.MOGA import MOGA
from model.MPGA import MPGA
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Read file by URL
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat"
data = pd.read_csv(url, header=None, sep = " ", )
FullFeatures = ['Att_'+str(i) for i in range(1, data.shape[1])]
TargetFeature = ['Target']
data.columns = FullFeatures + TargetFeature
scaler = MinMaxScaler()
scaler.fit(data[FullFeatures])
data[FullFeatures] = scaler.transform(data[FullFeatures])

# 1) GA

config_GA = {'population_size': 100, 'offspring_ratio': 0.5,
'crossover_probability': 0.9,
'selection_method': {'type': 'roulette_wheel', 'k': 10},
'crossover_method': {'type': '1point', 'parameters': None},
'mutation_probability': 0.1, 'mutation_ratio': 0.2,
'generations_number': 15, 'stop_criterion_depth': 100}


config_init_GA = {
    'FullFeatures': FullFeatures,
    'TargetFeature': TargetFeature,
    'Table': data,
    'ScoreType': 'accuracy_score',
    'InitMethod': 'random_init'
}

population_GA = Population.population_initialization(config_init_GA, new_population = None, population_size = 100)
solution_GA, fitness_GA, all_best_fitness_GA = population_GA.generate_populations(config=config_GA, verbose=1)
print(fitness_GA)
print(len(all_best_fitness_GA))
print(solution_GA.FeatureSubset)



# 2) MPGA
config_MPGA = {'population_size': 100, 'offspring_ratio': 0.5,
'crossover_probability': 0.9,
'selection_method': {'type': 'roulette_wheel', 'k': 10},
'crossover_method': {'type': '1point', 'parameters': None},
'mutation_probability': 0.1, 'mutation_ratio': 0.2,
'generations_number': 15, 'stop_criterion_depth': 100}


config_init_MPGA = {
    'FullFeatures': FullFeatures,
    'TargetFeature': TargetFeature,
    'Table': data,
    'ScoreType': 'accuracy_score',
    'InitMethod': 'random_init'
}
population_MPGA = MPGA.gas_initialization(config_init_MPGA, population_size = 100, gas_size=3)
best_solution_MPGA, best_fitness_MPGA, all_best_fitness_MPGA = population_MPGA.generate_gas(config_MPGA)
print(best_fitness_MPGA)
print(len(all_best_fitness_MPGA))


# 3) MOGA
config_init_MOGA = {
    'FullFeatures': FullFeatures,
    'TargetFeature': TargetFeature,
    'Table': data,
    'ScoreType': ['accuracy_score', 'number_feature_subset'],
    'InitMethod': 'random_init'
}

R = MOGA.population_initialization(config_init_MOGA, population_size = 100)


config_MOGA = {'population_size': 100, 'offspring_ratio': 0.5,
'crossover_probability': 0.9,
'selection_method': {'type': 'roulette_wheel', 'k': 2},
'crossover_method': {'type': '1point', 'parameters': None},
'mutation_probability': 0.1, 'mutation_ratio': 0.1,
'generations_number': 5, 'stop_criterion_depth': 100}

population_MOGA = R.generate_populations(config_MOGA)

config_GA = {'population_size': 100, 'offspring_ratio': 0.5,
'crossover_probability': 0.9,
'selection_method': {'type': 'roulette_wheel', 'k': 10},
'crossover_method': {'type': '1point', 'parameters': None},
'mutation_probability': 0.1, 'mutation_ratio': 0.2,
'generations_number': 15, 'stop_criterion_depth': 100}


config_init_GA = {
    'FullFeatures': FullFeatures,
    'TargetFeature': TargetFeature,
    'Table': data,
    'ScoreType': 'accuracy_score',
    'InitMethod': 'random_init'
}

population_MOGA = Population.population_initialization(config_init_GA, new_population = population_MOGA, population_size = 100)
solution_MOGA, fitness_MOGA, all_best_fitness_MOGA = population_MOGA.generate_populations(config=config_GA, verbose=1)
#print(fitness_GA)
#print(len(all_best_fitness_GA))
#print(solution_GA.FeatureSubset)

# plot
x = [i for i in range(len(all_best_fitness_GA))]
import matplotlib.pyplot as plt
with plt.style.context('Solarize_Light2'):
    plt.plot(x, all_best_fitness_GA, 'o-', label='GA')
    plt.plot(x, all_best_fitness_MPGA, 'o-', label='MPGA')
    plt.plot(x, all_best_fitness_MOGA, 'o-', label='MOGA')
    plt.title('Evolutionary processes of GA, MPGA, and MOGA (Australia credit data set)')
    plt.xlabel('generations', fontsize=9)
    plt.ylabel('accuracy', fontsize=9)

plt.legend(loc='lower right')
plt.show()
