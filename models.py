
import sys
import os
sys.path.append('/cluster/home/rodamian/rt_prediction')
sys.path.append('/Users/damiano/Desktop/Project_ML/rt_prediction')

from python.analysis.experiment import Experiment, MetaboliteDB
from python.analysis.constants import exp_names, path_db
import pandas as pd
from xgboost import XGBRegressor, DMatrix, cv
import copy
import sklearn.linear_model as lm
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold, cross_validate
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import maxabs_scale
import pickle
from sklearn.svm import LinearSVR
from scipy.stats import uniform
from scipy.stats import randint
from python.analysis.constants import root
import multiprocessing as mp
import glob
import numpy as np


def train_model(X, y, params, model, scale=True, n_jobs=10, cv=3, n_params=10, ret_score=False):

    # Feature scaling
    if scale:
        X = maxabs_scale(X)
    else:
        X = X.apply(lambda x: pd.to_numeric(x, errors='coerce'))

    if len(X) < cv:
        cv = len(X)

    crossv_mod = copy.deepcopy(model)
    ret_mod = copy.deepcopy(model)

    grid = RandomizedSearchCV(model, params, cv=cv, scoring='neg_median_absolute_error', verbose=0, n_jobs=n_jobs,
                              n_iter=n_params, refit=False)
    grid.fit(X, y)

    cv_pred = KFold(n_splits=cv)

    # Use the same parameters for the training set to get CV predictions
    crossv_mod.set_params(**grid.best_params_)
    preds = cross_val_predict(crossv_mod, X=X, y=y, cv=cv_pred, n_jobs=n_jobs, verbose=0)
    s = cross_validate(crossv_mod, X=X, y=y, cv=cv_pred, n_jobs=n_jobs, verbose=0, return_train_score=True)
    scores = [np.mean(s["train_score"]), np.mean(s["test_score"])]

    # Train the final model
    ret_mod.set_params(**grid.best_params_)
    ret_mod.fit(X, y)

    if ret_score:
        return scores

    return ret_mod, preds


class MLModel:

    # An ML model is just designed as a container for an ML experiment
    def __init__(self):
        self.parameters = {}

    def __setitem__(self, key, value):
        if key in self.parameters:
            self.parameters[key] = value
        else:
            print("Key not found:", key)

    def set_parameters(self, params):
        for pp in params.keys():
            self[pp] = params[pp]

    def list_parameters(self):
        return self.parameters.keys()


class XGBoostModel(MLModel):
    def __init__(self):
        super().__init__()
        self.parameters = {'n_estimators': randint(10, 200), 'max_depth': randint(1, 12),
                           'learning_rate': uniform(0.01, 0.25), 'gamma': uniform(0.0, 10.0),
                           'reg_alpha': uniform(0.0, 10.0), 'reg_lambda': uniform(0.0, 10.0)}

        self.model = XGBRegressor(**self.parameters)
        self.model_cv = None

    def fit(self, data=None, values=None, dm=None, metric="median_absolute_error"):
        if dm is None:
            dm = DMatrix(data, label=values)
            self.model.fit(data, values)
            self.model_cv = cv(params=self.parameters, dtrain=dm)

    def get_metrics(self):
        if self.model_cv is None:
            raise Exception("Model should be computed first")
        return self.model_cv.tail(1)

    def get_importance(self):
        return self.model.feature_importances_


class LassoModel(MLModel):
    def __init__(self):
        super().__init__()
        self.parameters = {'alpha': uniform(0.0, 10.0), 'copy_X': [True],
                           'fit_intercept': [True, False], 'normalize': [True, False],
                           'precompute': [True, False], 'max_iter': [1000000], 'tol': [0.0001]}

        self.model = lm.Lasso(**self.parameters)
        self.model_cv = None

    def fit(self, data=None, values=None, dm=None, metric="median_absolute_error"):
        if dm is None:
            dm = DMatrix(data, label=values)
            self.model.fit(data, values)
            self.model_cv = cv(params=self.parameters, dtrain=dm)

    def get_metrics(self):
        if self.model_cv is None:
            raise Exception("Model should be computed first")
        return self.model_cv.tail(1)

    def get_importance(self):
        return self.model.coef_


class SVRModel(MLModel):
    def __init__(self):
        super().__init__()
        self.parameters = {'C': uniform(0.01, 300.0), 'dual': [True],
                           'epsilon': uniform(0.01, 300.0), 'fit_intercept': [True, False],
                           'intercept_scaling': uniform(0, 1.0),
                           'loss': ['squared_epsilon_insensitive', 'epsilon_insensitive'], 'max_iter': [100000],
                           'tol': [0.001], 'verbose': [0]}

        self.model = LinearSVR(**self.parameters)
        self.model_cv = None

    def fit(self, data=None, values=None, dm=None, metric="median_absolute_error"):
        if dm is None:
            dm = DMatrix(data, label=values)
            self.model.fit(data, values)
            self.model_cv = cv(params=self.parameters, dtrain=dm)

    def get_metrics(self):
        if self.model_cv is None:
            raise Exception("Model should be computed first")
        return self.model_cv.tail(1)

    def get_importance(self):
        return self.model.coef_


if __name__ == "__main__":

    db = MetaboliteDB(path_db)

    def get_imp(exp_name, db=db):
        importance = pd.DataFrame()
        pred_error = pd.DataFrame()

        exp = Experiment(exp_name, db)
        features, rt = exp.get_features()
        print(exp.name, len(exp.features))

        for m in [LassoModel, XGBoostModel, SVRModel]:
            model = m()
            trained_mod, pred = train_model(features, rt.values.ravel(), model.parameters, model.model, n_jobs=1)
            model.model = trained_mod
            pred_error[m.__name__] = abs((rt.values.ravel() - pred) / max(rt.values.ravel()))
            importance[m.__name__] = maxabs_scale(model.get_importance())
        importance.set_index(features.columns, inplace=True)

        # Writing dataframe to files
        if not os.path.exists(os.path.join(root, "data", "prediction_error")):
            os.makedirs(os.path.join(root, "data", "prediction_error"))

        with open(os.path.join(root, "data", "prediction_error", "p_" + exp_name), "wb") as p:
            pickle.dump(pred_error, p)
        p.close()

        if not os.path.exists(os.path.join(root, "data", "f_importance")):
            os.makedirs(os.path.join(root, "data", "f_importance"))

        with open(os.path.join(root, "data", "f_importance", "f_" + exp_name), "wb") as f:
            pickle.dump(importance, f)
        f.close()

    with mp.Pool(mp.cpu_count()) as p:
        p.map(get_imp, exp_names)

    # Merging temporary databases
    with open(os.path.join(root, "data", "feature_total"), "wb") as f_total:
        tot = pd.DataFrame()
        for f in glob.glob(os.path.join(root, "data", "f_importance", "f_*")):
            temp = pickle.load(open(f, "rb"))
            temp["experiment"] = f[f.rindex("/f_") + 3:]
            temp.set_index("experiment", append=True, inplace=True)
            tot = tot.append(temp)
        pickle.dump(tot, f_total)
    f_total.close()

    with open(os.path.join(root, "data", "prediction_total"), "wb") as p_total:
        tot = pd.DataFrame()
        for f in glob.glob(os.path.join(root, "data", "prediction_error", "p_*")):
            temp = pickle.load(open(f, "rb"))
            temp["experiment"] = f[f.rindex("/p_") + 3:]
            temp.set_index("experiment", append=True, inplace=True)
            tot = tot.append(temp)
        pickle.dump(tot, p_total)
    p_total.close()
