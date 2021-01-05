
import pandas as pd
from rdkit.ML.Descriptors import MoleculeDescriptors
from rdkit import Chem
import os
import pickle
from python.analysis.constants import descList, columns, path_db, data_path, exp_names
import xgboost as xgb
import numpy as np
from sklearn.feature_selection import VarianceThreshold


class MetaboliteDB:

    # MetaboliteDatabase class, which contains a list of precomputed descriptors
    def __init__(self, path_db, buffer=50):
        self.path_db = path_db
        if os.path.isfile(path_db):
            db = pickle.load(open(path_db, "rb"))
            self.metabolites = db.metabolites
        else:
            self.metabolites = {}
        self.counter_added = 0
        self.buffer = buffer

    def __getitem__(self, id):
        if id in self.metabolites:
            return self.metabolites[id]
        else:
            self.add_metabolite(id)
        return self.metabolites[id]

    def add_metabolite(self, id):

        # Add metabolite function which casts strings of identifiers in the class Metabolite and computes its descriptors
        self.counter_added += 1
        if self.counter_added == self.buffer:
            self.save_db()
            self.counter_added = 0

        if id != id:
            self.metabolites[id] = 0

        m = Metabolite(id)
        self.metabolites[id] = m.get_descriptors()

    def save_db(self):
        with open(path_db, "wb") as database:
            pickle.dump(self, database)

    def reset_counter(self):
        self.counter_added=0


class Analysis:

    # Analysis takes as inputs a list of Experiment objects and is used to return feature importances and select features
    def __init__(self, experiments):
        self.columns = [columns[exp.name]["columns"] for exp in experiments]
        self.experiments = experiments
        self.metabolites = [exp.metabolites for exp in experiments]
        self.rts = [exp.rts for exp in experiments]

    def average_imp(self):
        sum_imp = np.zeros(len(descList))
        for exp in self.experiments:
            features = pd.DataFrame(exp.metabolites)
            features = features.loc[:, (features != features.iloc[0]).any()]
            m = xgb.XGBRegressor().fit(features, exp.rts)
            sum_imp += m.feature_importances_
        avg_imp = sum_imp/len(self.experiments)

        return avg_imp


class Metabolite:

    # Metabolite class has attributes struct which represents structure in the rdkit toolkit
    # Function get_descriptors gets called when a Metabolite object gets called by the Experiment class and computes its descriptors
    def __init__(self, char):
        self.struct = None
        if type(char) is str:
            if char.startswith("InChI"):
                self.struct = Chem.inchi.MolFromInchi(char)
            else:
                self.struct = Chem.MolFromSmiles(char)

    def get_descriptors(self):

        # Calculate descriptors from metabolite structure
        if self.struct is not None:
            temp = MoleculeDescriptors.MolecularDescriptorCalculator(descList)
            return temp.CalcDescriptors(self.struct)


class Experiment:

    def __init__(self, id_exp, db=None, path=data_path):

        self.name = id_exp
        if db is None:
            db = MetaboliteDB(path_db)

        data = pd.read_table(path, low_memory=False)
        data.dropna(subset=["smiles", "inchi"], how="all", inplace=True)
        data = data[data.id_experiment == id_exp]
        data.sort_values(by="rt", inplace=True)
        data.reset_index(drop=True, inplace=True)

        # Look for metabolite in DB
        self.features = pd.DataFrame([db[m] for m in data.inchi])
        self.features["rt"] = data.rt
        self.features.dropna(subset=["rt"])
        self.features.dropna(how="all", inplace=True)
        self.features = self.features[self.features.nunique(axis=1) != 1].dropna()

        self.rts = self.features["rt"]
        self.features.drop("rt", axis=1, inplace=True)
        self.features.columns = descList[0:len(self.features.columns)]

        # drop quasi constant features
        constant_filter = VarianceThreshold(threshold=0.005)
        constant_filter.fit(self.features.values)
        constant_columns = [column for column in self.features.columns
                            if column not in self.features.columns[constant_filter.get_support()]]

        self.features.drop(labels=constant_columns, axis=1, inplace=True)

        # Create correlation matrix
        corr_matrix = self.features.corr().abs()

        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # Find features with correlation greater than 0.95
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

        # Drop features
        self.features.drop(to_drop, axis=1, inplace=True)

        self.columns = columns[id_exp]["columns"] if id_exp in columns else None

        self.met_name = data.name

    def get_features(self):
        return self.features, self.rts
