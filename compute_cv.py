
import sys
import os
sys.path.append('/cluster/home/rodamian/rt_prediction')
sys.path.append('/Users/damiano/Desktop/Project_ML/rt_prediction')

import pandas as pd
import pickle
from python.analysis.models import XGBoostModel, train_model
import os
from python.analysis.constants import root, exp_names, data_path, exp_c, lengths
from python.analysis.experiment import Experiment, MetaboliteDB, path_db
import glob
import multiprocessing as mp
data = pd.read_table(data_path, low_memory=False)


def get_cv_score(id=None, sel_feat=None, db=None, n_feat=30, n_params=50):

    if db is None:
        db = MetaboliteDB(path_db)
    exp = Experiment(id, db)

    if sel_feat is None:
        top_f = pickle.load(open(os.path.join(root, "data", "sorted_feat"), "rb"))
        # sel_feat_for_col = top_f.iloc[:, [exp.name in exp_c[col] for col in top_f]]
        sel_feat = top_f["all"].tolist()

    m = XGBoostModel()
    features, rt = exp.get_features()
    cv_ratio = pd.DataFrame(columns=[exp.name], index=["train", "test"])

    # Compute CV error for first n features all columns
    for i, feat in enumerate(sel_feat[0:n_feat]):
        current = sel_feat[0:(i+1)]

        if features.columns.isin(current).any() is False:
            current = sel_feat[0:(i + 2)]

        print(current)
        cv_ratio[i] = train_model(features[features.columns & current], rt.values.ravel(),
                                  m.parameters, m.model, n_params=n_params, n_jobs=1, ret_score=True)
    cv_ratio = cv_ratio.transpose()
    cv_ratio["ratio"] = cv_ratio["test"]/cv_ratio["train"]

    # Check if directory exists otherwise create it
    if not os.path.exists(os.path.join(root, "data", "cv")):
        os.makedirs(os.path.join(root, "data", "cv"))

    with open(os.path.join(root, "data", "cv", "c_" + id), "wb") as f:
        pickle.dump(cv_ratio, f)
        f.close()


if __name__ == "__main__":

    # Data cleaning
    feature_importance = pickle.load(open(os.path.join(root, "data/feature_total"), "rb"))
    feature_importance.fillna(0, inplace=True)
    feature_importance.index.rename(names=["features", "experiment"], inplace=True)
    feature_importance.index = feature_importance.index.swaplevel()

    # Sum feature importance over experiment
    feature_over_exp = feature_importance.groupby("features").apply(lambda x: x.abs().sum())

    # Sum feature importance over model
    feature_over_model = feature_over_exp.sum(axis=1)

    # Select top n features
    sorted_feat = pd.DataFrame(feature_over_model.sort_values(ascending=False).index.values, columns=["all"])

    # top features for different columns
    for col in exp_c:

        summed_imp = feature_importance.loc[exp_c[col]].groupby("features").apply(lambda x: x.abs().sum())
        feat_names = pd.DataFrame(summed_imp.sum(axis=1).sort_values(ascending=False).index.values, columns=[col])
        sorted_feat[col] = feat_names

    print(sorted_feat[0:20])

    with open(os.path.join(root, "data", "sorted_feat"), "wb") as f:
        pickle.dump(sorted_feat, f)
    f.close()

    with mp.Pool(mp.cpu_count()) as p:
        p.map(get_cv_score, exp_names)

    # Merging temporary databases
    with open(os.path.join(root, "data", "cv_total"), "wb") as c_total:
        cv_tot = pd.DataFrame()
        for f in glob.glob(os.path.join(root, "data", "cv", "c_*")):
            temp = pickle.load(open(f, "rb"))
            cv_tot = cv_tot.append(temp)

        pickle.dump(cv_tot, c_total)
    c_total.close()
