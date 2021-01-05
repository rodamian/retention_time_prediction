
import matplotlib.pylab as plt
import pandas as pd
import pickle
import os
from python.analysis.constants import root
from python.analysis.experiment import Experiment
import numpy as np
from python.analysis.experiment import MetaboliteDB, path_db
db = MetaboliteDB(path_db)
plt.style.use("ggplot")

# Load prediction error data
prediction_error = pickle.load(open(os.path.join(root, "data", "prediction_total"), "rb"))

# Plot number of met for experiment
prediction_error.groupby("experiment").size().plot.hist(bins=np.arange(0, 800, 20))
plt.ylabel("experiment count")
plt.show()

# Plots % of prediction under threshold
for t in range(1, 16):
    under_t = prediction_error.apply(lambda x: (x < t/100)).sum() / len(prediction_error) * 100
    plt.bar(x=[t, t + 0.2, t + 0.4], height=under_t, align="center", width=0.18, color=["C7", "C8", "C9"])
plt.legend(["Lasso", "XGBoost", "SVR"])
plt.ylabel("% under threshold", fontsize=20)
plt.xlabel("realative error (%)", fontsize=20)
plt.ylim(0, 80)
plt.show()

# Plotting mean prediction error vs. number metabolites
for m in prediction_error[["XGBoostModel", "LassoModel", "SVRModel"]]:
    plt.scatter(x=prediction_error[m].groupby("experiment").count(),
                y=prediction_error[m].groupby("experiment").mean(),
                label=m)
plt.xlabel("Metabolites for experiment", fontsize=15)
plt.ylabel("Mean prediction error %", fontsize=15)
plt.scatter(79907, 0.1046520, label="METLIN", edgecolors="grey")
plt.scatter(79907, 0.11434850, label="METLIN", edgecolors="grey")
plt.xscale("log")
plt.ylim(0, 0.7)
plt.xlim(10)
plt.legend()
plt.show()

# Plotting mean prediction error vs. number metabolites for Lasso and XGBoost with error bar
pos = prediction_error.groupby("experiment").size().sort_values().to_list()
w = 0.02
width = lambda p, w: 10**(np.log10(p)+w/2.)-10**(np.log10(p)-w/2.)
prediction_error.boxplot(column=["LassoModel", "XGBoostModel"], by="experiment",
                         positions=prediction_error.groupby("experiment").size().sort_values().to_list(),
                         showfliers=False, widths=width(pos, w),
                         boxprops=dict(linestyle='-', linewidth=1),
                         medianprops=dict(linestyle='solid', linewidth=2, color="r"))

plt.xlabel("Metabolites for experiment")
plt.ylabel("Mean prediction error %")
plt.xscale("log")
plt.ylim(0, 1)
plt.show()


# Plot selected datasets for comparison with paper
plt.style.use(("ggplot"))
rt = [Experiment(exp).rts.max() for exp in ["Stravs", "FEM_long", "Taguchi_12", "Eawag_XBridgeC18", "RIKEN"]]
sel = prediction_error.groupby('experiment').apply(lambda x: x.sample(50, replace=True).median()).loc[[
                "Stravs", "FEM_long", "Taguchi_12", "Eawag_XBridgeC18", "RIKEN"]]
sel = sel.mul(rt, axis=0)
comp = pd.DataFrame(data=np.array([[83.4, 79, 82], [250, 150, 248], [110, 96, 95], [82, 79, 85], [14, 8, 13]]),
                    index=sel.index, columns=sel.columns)

sel = sel.append(comp)
sel.plot.bar()
ax = sel.plot.bar()
plt.ylabel("Median absolute error", fontsize=12)
plt.xticks(rotation=45, fontsize=7)
secax = ax.secondary_xaxis('top')
secax.set_xlabel('Damiano      Bouwmeester')
plt.legend()
plt.show()

# Plot annotation improvement
plt.style.use("ggplot")
annotation = pd.DataFrame(pickle.load(open(os.path.join(root, "data", "annotation_improvement_mtbls127_100feat"), "rb")))
annotation.boxplot()
plt.ylabel("correctly identified metabolites", fontsize=10)
plt.xlabel("metabolites used for training", fontsize=10)

# Plot rt density of validation datasets
for name_exp in ["mtbls127_annotation.csv", "mtbls586_annotation.csv",
                 "mtbls631_annotation.csv", "mtbls55_annotation.csv"]:

    data = pd.read_csv(os.path.join(root, "validation_data", name_exp), sep=";")
    data["rt"].plot.density(label=name_exp.split("_")[0], bw_method=0.1)
plt.legend()
plt.xlim(0, 30)
plt.xlabel("rt (min)")
plt.show()
