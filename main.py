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

#np.random.seed(0)

config = {'population_size': 100, 'offspring_ratio': 0.5,
'crossover_probability': 0.5,
'selection_method': {'type': 'roulette_wheel', 'k': 10},
'crossover_method': {'type': '1point', 'parameters': None},
'mutation_probability': 0.1, 'mutation_ratio': 0.1,
'generations_number': 20, 'stop_criterion_depth': 20}


config_init = {
    'FullFeatures': FullFeatures,
    'TargetFeature': TargetFeature,
    'Table': data,
    'ScoreType': 'accuracy_score',
    'InitMethod': 'random_init'
}

population = Population.population_initialization(config_init, population_size = 100)
solution, fitness, all_best_fitness = population.generate_populations(config=config, verbose=1)
