import numpy as np
import pandas as pd
import ranking_single_filter as SF

# Read file by URL
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/australian/australian.dat"
data = pd.read_csv(url, header=None, sep = " ", )
FullFeatures = ['Att_'+str(i) for i in range(1, data.shape[1])]
TargetFeature = ['Target']
data.columns = FullFeatures + TargetFeature

# Using ranking - filter
congig_filer = {
    'FullFeatures': FullFeatures,
    'TargetFeature': TargetFeature,
    'Table': data,
    'FilterMethod':{'type':'ChiSquare_method', 'k': 'all'}
}
SingleFilter = SF.filter_single_approach(congig_filer)
print(SingleFilter.fit())
