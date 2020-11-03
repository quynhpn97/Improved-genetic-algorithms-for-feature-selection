import numpy as np
import pandas as pd

class filter_group_approach():
    def __init__(self, ConfigFilter):
        self.ConfigFilter = ConfigFilter
        # Check conditions for ConfigFilter
        if check(self.ConfigFilter):
            self.FullFeatures = self.ConfigFilter['FullFeatures']
            self.FilterMethod = self.ConfigFilter['FilterMethod']
            self.Table = self.ConfigFilter['Table']
            self.TargetFeature = self.ConfigFilter['TargetFeature']
    def best_first_search(self):
        open = []
        closed = []
        pass

    @classmethod
    def continous_class_merit(self, FeatureSubset):
        X = self.Table[FeatureSubset]
        y = self.Table[self.TargetFeature]
        matrix_corr = np.corrcoef(pd.concat([X, y], axis = 1).values)
        R_cf = np.mean(matrix_corr[:-1][-1])
        R_ff = matrix_corr[np.triu_indices_from(matrix_corr,1)].mean()
        k = len(FeatureSubset)
        return k*R_cf/np.sqrt(k+k*(k-1)*R_ff)

    @classmethod
    def discrete_class_merit(self, FeatureSubset):
        pass
