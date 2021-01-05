
import sys
import os
sys.path.append('/cluster/home/rodamian/rt_prediction')
sys.path.append('/Users/damiano/Desktop/Project_ML/rt_prediction')

import matplotlib.pylab as plt
import pandas as pd
import pickle
import os
from python.analysis.constants import root, lengths, selected_features, exp_cols, exp_c
from python.analysis.experiment import Experiment, MetaboliteDB, path_db, descList
from sklearn.preprocessing import maxabs_scale
db = MetaboliteDB(path_db)
plt.style.use("ggplot")

if __name__ == "__main__":

    # Data cleaning
    feature_importance = pickle.load(open(os.path.join(root, "data/feature_total"), "rb"))
    feature_importance.fillna(0, inplace=True)
    feature_importance.index.rename(names=["features", "experiment"], inplace=True)
    feature_importance.index = feature_importance.index.swaplevel()

    # Sum feature importance over experiment
    feature_over_exp = feature_importance.groupby("features").apply(lambda x: x.abs().sum())

    # Plot feature importance
    feature_over_exp[["LassoModel", "XGBoostModel"]].groupby("features").apply(lambda x: x.abs().sum()).plot()

    # Sum feature importance over model
    feature_over_model = feature_over_exp.sum(axis=1)

    # Select top n features
    sorted_feat = pd.Series(feature_over_model.sort_values(ascending=False))
    print(sorted_feat[0:10])

    feat_for_col = pd.DataFrame()
    top_n = pd.DataFrame()

    # top features for different columns
    for col in exp_c:

        summed_imp = feature_importance.loc[exp_c[col]].groupby("features").apply(lambda c: c.abs().sum())
        summed_imp = summed_imp.apply(maxabs_scale)
        feat_for_col[col] = summed_imp.sum(axis=1)

        top_n[col] = feat_for_col[col].sort_values(ascending=False)[0:10].index

    print(top_n)

    # Prediction error analysis
    prediction_error = pickle.load(open(os.path.join(root, "data/prediction_total"), "rb"))
    prediction_mean = prediction_error.groupby("experiment").mean()

    # Load merged databases
    cv = pickle.load(open(os.path.join(root, "data", "cv_total"), "rb"))
    cv = cv.abs().transpose().apply(maxabs_scale)
    cv.plot()
    plt.show()

    # Plot prediction error decrease with increasing number of features
    for n in (50, 80, 100000):
        cv[cv.columns & lengths.keys()[lengths < n]].abs().mean(axis=1).plot(label="< " + str(n))

    plt.xlabel("number of features added", fontsize=20)
    plt.ylabel("normalized mean absolute error", fontsize=20)
    plt.xlim(-1, 29)
    plt.ylim(0.5, 1)
    plt.legend()
    plt.show()

cv_tot = pickle.load(open(os.path.join(root, "data", "cv_total"), "rb"))
cv_tot_clean = cv_tot["test"]
cv_tot_clean.dropna(inplace=True)

plt.plot(cv_tot_clean.abs())
plt.ylim(0, 10)
plt.xlim(-1, 30)
plt.show()
