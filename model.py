import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.base import TransformerMixin
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestClassifier

class CustomMixin(TransformerMixin):
    def get_params(self, **kwargs):
        return dict()

    def set_params(self, **kwargs):
        for key in self.get_params():
            setattr(self, key, kwargs[key])

class ReplaceOutliers(CustomMixin):
    def fit(self, X, y):
        self.replace_value = X.YearMade[X.YearMade > 1900].mode()
        return self

    def transform(self, X):
        condition = X.YearMade > 1900
        X.loc[:,'YearMade_imputed'] = 0
        X.loc[~condition, 'YearMade_imputed'] = 1
        X.loc[~condition, 'YearMade'] = self.replace_value[0]
        return X

class CalculateAge(CustomMixin):
    def fit(self, X, y):
        return self

    def transform(self, X):
        X.SaleYear = X.saledate.map(lambda x: x[-9:-5]).astype(int)
        X.loc[:,'Age'] = (X.SaleYear - X.YearMade).astype(int)
        X.loc[:, 'SaleYear'] = X.SaleYear - X.SaleYear.min()
        return X

class ColumnFilter(CustomMixin):
    def __init__(self, cols=[]):
        self.cols = cols

    def get_params(self, **kwargs):
        return {'cols': self.cols}

    def fit(self, X, y):
        # Get the order of the index for y.
        return self

    def transform(self, X):
        X = X[self.cols].sort_index()
        return X

class Dummify(CustomMixin):
    def fit(self, X,y):
        return self

    def transform(self, X):
        X = pd.get_dummies(X)
        return X

class GroupEnclosure(CustomMixin):
    def fit(self, X,y):
        return self

    def transform(self, X):
        X.loc[:,'Enclosure_Group'] = 'None' # nan, no rops, none or unspecified
        X.loc[X.Enclosure.isin(['EROPS', 'EROPS w AC', 'EROPS AC']), 'Enclosure_Group'] = 'EROPS'
        X.loc[X.Enclosure == 'OROPS', 'Enclosure_Group'] = 'OROPS'
        return X

class GroupHydraulics(CustomMixin):
    def fit(self, X,y):
        return self

    def transform(self, X):
        X.loc[:,'Hydraulics_Group'] = 'None' # nan, no rops, none or unspecified
        X.loc[X.Hydraulics.isin(['Auxiliary', 'Standard']), 'Hydraulics_Group'] = 'AuxSta'
        X.loc[X.Hydraulics.str.contains('Valve', na = False), 'Hydraulics_Group'] = 'Valve'
        X.loc[X.Hydraulics.str.contains('Base', na = False), 'Hydraulics_Group'] = 'Base'
        return X

def prep_train_test_set(train_in, test_in, test_result):
    df_train = pd.read_csv(train_in).set_index('SalesID').sort_index()
    df_test = pd.read_csv(test_in).set_index('SalesID').sort_index()

    X_train = df_train.drop('SalePrice', axis=1)
    X_test = df_test.copy(deep=True)
    y_train = df_train.SalePrice
    y_test = pd.read_csv(test_result).set_index('SalesID').SalePrice

    return X_train, y_train, X_test, y_test

def rmsle(y_pred, y_true):
    log_diff = np.log(y_pred+1) - np.log(y_true+1)
    return np.sqrt(np.mean(log_diff**2))

if __name__ == '__main__':

    X_train, y_train, X_test, y_test = prep_train_test_set('data/Train.csv', 'data/Test.csv','data/do_not_open/test_soln.csv')

    p = Pipeline([
        # ('type_change', DataType()),
        ('replace_outliers', ReplaceOutliers()),
        ('calculate_age', CalculateAge()),
        ('group_enclosure', GroupEnclosure()),
        ('group_hydraulics', GroupHydraulics()),
        ('columns', ColumnFilter()),
        # ('impute', Imputer(copy=False)),
        # ('filter', FilterRows()),
        ('dummify', Dummify()),
        ('lm', LinearRegression(normalize=True,n_jobs=-1))
    ])
    params = {'columns__cols': [['Age', 'ProductGroup','Hydraulics_Group']]}

    # b = Pipeline(p.steps[:-1])

    scorer = make_scorer(rmsle, greater_is_better=False)
    gscv = GridSearchCV(p, params, scoring=scorer, cv=5)
    clf = gscv.fit(X_train, y_train)

    print 'Best parameters: %s' % clf.best_params_
    print 'Best RMSLE: %s' % clf.best_score_
    print 'Scores: %s' % clf.grid_scores_

    model = clf.best_estimator_
    y_pred = model.predict(X_test)
    print rmsle(y_pred, y_test)
