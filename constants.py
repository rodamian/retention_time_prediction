
from rdkit.Chem import Descriptors
import json
from pathlib import Path
import pandas as pd
import pickle
import os


def get_project_root() -> Path:
    # Returns project root folder
    return Path(__file__).parent.parent.parent


root = get_project_root()

# List of descriptors calculated by RDKit
descList = [dd[0] for dd in Descriptors.descList]

# List of Chromatographic systems used
path_columns = os.path.join(str(root), "data", "json_metabolights.json")
with open(path_columns) as f:
    columns = json.load(f)

# Paths to data table and descriptors Database
path_db = os.path.join(str(root), "data", "MetaboliteDB")
data_path = os.path.join(str(root), "data", "complete_dataset.tsv")
exp_names = list(pd.read_table(data_path,
                               dtype={"id": str, "formula": str, "smiles": str, "inchikey": str,
                                      "charge": str, "mz": str, "valid": str})["id_experiment"].unique())

data = pd.read_table(data_path)
lengths = pd.Series([len(data[data.id_experiment == exp]) for exp in exp_names], index=exp_names)
exp_cols = {exp: columns[exp]["columns"] for exp in exp_names if exp in columns}
exp_cols.update({exp: 'None' for exp in exp_names if exp not in columns})

# Merge column types for similarity
# Dictionary with keys: columns, values: experiment names

exp_c = {'RP|C18': [k for k, v in exp_cols.items() if 'RP|C18|' in v],
         'RP': [k for k, v in exp_cols.items() if 'RP|' in v and 'RP|C18|' not in v],
         'HILIC': [k for k, v in exp_cols.items() if 'HILIC|' in v]}

exp_c.update({"all": [k for k, v in exp_cols.items() if all(c not in v for c in [*exp_c])]})

selected_features = pickle.load(open(os.path.join(str(root), "data", "sorted_feat"), "rb"))
