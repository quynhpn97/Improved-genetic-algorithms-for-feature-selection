from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import f_classif
def check(congig_filer = dict()):
    pass
def CalculateFScore():
    pass
class filter_single_approach():

    def __init__(self, ConfigFilter):
            self.ConfigFilter = ConfigFilter
        # Check conditions for ConfigFilter
        #if check(self.ConfigFilter):
            self.FullFeatures = self.ConfigFilter['FullFeatures']
            self.FilterMethod = self.ConfigFilter['FilterMethod']
            self.Table = self.ConfigFilter['Table']
            self.TargetFeature = self.ConfigFilter['TargetFeature']
            if self.FilterMethod['k'] == 'all':
                self.K = len(self.FullFeatures)
            else:
                self.K = int(self.FilterMethod['k'])

    def LDA_method(self):
        clf = LinearDiscriminantAnalysis()
        clf.fit(self.Table[self.FullFeatures], self.Table[self.TargetFeature])
        # Get Weight vector(s)
        Coef = clf.coef_
        return Coef

    def F_method(self):
        F_score = {}
        for f_i in self.FullFeatures:
            F_score[f_i] = CalculateFScore(self.Table[f_i], self.Table[self.TargetFeature])

    def RST_method(self):
        pass
    def Gini_method(self):
        pass

    def ANOVA_method(self):
        fs = SelectKBest(score_func=f_classif, k='all')
        fs.fit(self.Table[self.FullFeatures], self.Table[self.TargetFeature])
        score = {}
        for f_i, _ in zip(self.FullFeatures, range(len(fs.scores_))):
            score[f_i] = fs.scores_[_]
        return score

    def MutualInfo_method(self):
        fs = SelectKBest(score_func=mutual_info_classif, k='all')
        fs.fit(self.Table[self.FullFeatures], self.Table[self.TargetFeature])
        score = {}
        for f_i, _ in zip(self.FullFeatures, range(len(fs.scores_))):
            score[f_i] = fs.scores_[_]
        return score

    def ChiSquare_method(self):
        fs = SelectKBest(score_func=chi2, k='all')
        fs.fit(self.Table[self.FullFeatures], self.Table[self.TargetFeature])
        score = {}
        for f_i, _ in zip(self.FullFeatures, range(len(fs.scores_))):
            score[f_i] = fs.scores_[_]
        return score

    def fit(self):
        # selection method phase
        MethodSwitcher = {
            'LDA_method': self.LDA_method,
            'F_method': self.F_method,
            'RST_method': self.RST_method,
            'Gini_method': self.Gini_method,
            'ANOVA_method': self.ANOVA_method,
            'MutualInfo_method': self.MutualInfo_method,
            'ChiSquare_method': self.ChiSquare_method
        }

        self.SelectedFeatures = MethodSwitcher.get(
                    self.FilterMethod['type'],
                    lambda: 'Invalid method')()
        return self.SelectedFeatures
