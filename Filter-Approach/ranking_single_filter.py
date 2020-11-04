<<<<<<< HEAD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
=======
from sklearn.lda import LDA
>>>>>>> 3ddd189e9473c773b15a31b55f0bd9c2426f4bc1
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import f_classif
def check(congig_filer = dict()):
    pass
def CalculateFScore():
    pass
<<<<<<< HEAD
class filter_single_approach():

    def __init__(self, ConfigFilter):
            self.ConfigFilter = ConfigFilter
        # Check conditions for ConfigFilter
        #if check(self.ConfigFilter):
=======
class filter_single_approach(object):

    def __init__(self, ConfigFilter):
        self.ConfigFilter = ConfigFilter
        # Check conditions for ConfigFilter
        if check(self.ConfigFilter):
>>>>>>> 3ddd189e9473c773b15a31b55f0bd9c2426f4bc1
            self.FullFeatures = self.ConfigFilter['FullFeatures']
            self.FilterMethod = self.ConfigFilter['FilterMethod']
            self.Table = self.ConfigFilter['Table']
            self.TargetFeature = self.ConfigFilter['TargetFeature']
            if self.FilterMethod['k'] == 'all':
                self.K = len(self.FullFeatures)
            else:
                self.K = int(self.FilterMethod['k'])

    def LDA_method(self):
<<<<<<< HEAD
        clf = LinearDiscriminantAnalysis()
        clf.fit(self.Table[self.FullFeatures], self.Table[self.TargetFeature])
        # Get Weight vector(s)
        Coef = clf.coef_
        return Coef
=======
        clf = LDA()
        clf.fit(self.Table[self.FullFeatures], self.Table[self.TargetFeature])
        # Get Weight vector(s)
        Coef = clf.coef_
>>>>>>> 3ddd189e9473c773b15a31b55f0bd9c2426f4bc1

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
<<<<<<< HEAD
        return score

=======
        #
>>>>>>> 3ddd189e9473c773b15a31b55f0bd9c2426f4bc1
    def MutualInfo_method(self):
        fs = SelectKBest(score_func=mutual_info_classif, k='all')
        fs.fit(self.Table[self.FullFeatures], self.Table[self.TargetFeature])
        score = {}
        for f_i, _ in zip(self.FullFeatures, range(len(fs.scores_))):
            score[f_i] = fs.scores_[_]
<<<<<<< HEAD
        return score
=======
>>>>>>> 3ddd189e9473c773b15a31b55f0bd9c2426f4bc1

    def ChiSquare_method(self):
        fs = SelectKBest(score_func=chi2, k='all')
        fs.fit(self.Table[self.FullFeatures], self.Table[self.TargetFeature])
        score = {}
        for f_i, _ in zip(self.FullFeatures, range(len(fs.scores_))):
            score[f_i] = fs.scores_[_]
<<<<<<< HEAD
        return score
=======
>>>>>>> 3ddd189e9473c773b15a31b55f0bd9c2426f4bc1

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
<<<<<<< HEAD
                    lambda: 'Invalid method')()
=======
                    lambda: 'Invalid method')(
                        self.FilterMethod['k']
                    )
>>>>>>> 3ddd189e9473c773b15a31b55f0bd9c2426f4bc1
        return self.SelectedFeatures
