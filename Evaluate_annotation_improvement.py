
from python.analysis.experiment import MetaboliteDB
from python.analysis.constants import path_db, descList
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import pandas as pd
import pickle
import random
from python.analysis.constants import selected_features, root
import os
from xgboost import XGBRegressor

name_exp = "mtbls127_annotation.csv"
iterations = 10

annotation = pd.read_csv(os.path.join(root, "validation_data", name_exp), sep=";")

db = MetaboliteDB(path_db)
features = pd.DataFrame([db[m] for m in annotation.smiles])
features.columns = descList[0:len(features.columns)]
rts = annotation.rt

# drop quasi constant features
constant_filter = VarianceThreshold(threshold=0.01)
constant_filter.fit(features.values)
constant_columns = [column for column in features.columns
                    if column not in features.columns[constant_filter.get_support()]]

features.drop(labels=constant_columns, axis=1, inplace=True)

# Drop correlated features
# Select upper triangle of correlation matrix
corr_matrix = features.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Drop features
features.drop(to_drop, axis=1, inplace=True)

# Create dictionary with keys: sampling used, values: correctly identified metabolites
results = dict()

sampling = [20, 50, 80, len(features)]

for sampled in sampling:

    correct = []
    for i in range(iterations):

        # Sample n metabolites from dataset
        f_sampled = features.sample(sampled)
        r_sampled = rts[f_sampled.index]

        # Use only top n features
        n_feat = 100
        f_sampled = f_sampled[f_sampled.columns & selected_features["all"].to_list()[0:n_feat]]

        # Train model with sampled metabolites
        model = XGBRegressor().fit(f_sampled, r_sampled)
        # Add predictions for candidates
        pred_rt = []
        for s in annotation.index:
            candidate_feat = pd.DataFrame([db[m] for m in annotation.candidates_smiles.iloc[s].split('|')])
            candidate_feat.columns = descList[0:len(candidate_feat.columns)]
            candidate_feat = candidate_feat[candidate_feat.columns & f_sampled.columns]

            # Predicting candidate rts with model trained
            pred = model.predict(candidate_feat)
            pred_rt.append('|'.join(map(str, pred)))

        # Add predicted rt to dataframe
        annotation["pred_rt"] = pred_rt

        best_match = []
        for s in annotation.index:
            best_match.append(np.argmin(np.abs(np.asarray(annotation.pred_rt[s].split('|'),
                                                          dtype=np.float) - annotation.rt[s])) + 1)

        annotation["best_match"] = best_match
        correct.append((annotation["best_match"] == annotation["true_annot"]).sum())

    results[sampled] = correct
    print(np.mean(correct), "correctly annotated on",
          len(annotation), "masses with multiple candidates with", sampled, "sampled for training")

results["random"] = [([random.randint(1, c) for c in annotation.num_candidates] == annotation.true_annot).sum()
                     for i in range(iterations)]

with open(os.path.join(root, "data", "annotation_improvement_" + name_exp.split("_")[0] + "_" + str(n_feat) + "feat"), "wb") as f:
    pickle.dump(results, f)
