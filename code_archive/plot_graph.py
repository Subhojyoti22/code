# Imports and defaults
import joblib
from joblib import Parallel, delayed
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog
import time

import ipdb
import pickle

mpl.style.use("classic")
mpl.rcParams["figure.figsize"] = [6, 4]

mpl.rcParams["axes.linewidth"] = 0.75
mpl.rcParams["errorbar.capsize"] = 3
mpl.rcParams["figure.facecolor"] = "w"
mpl.rcParams["grid.linewidth"] = 0.75
mpl.rcParams["lines.linewidth"] = 0.75
mpl.rcParams["patch.linewidth"] = 0.75
mpl.rcParams["xtick.major.size"] = 3
mpl.rcParams["ytick.major.size"] = 3

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.titlesize"] = "medium"
mpl.rcParams["legend.fontsize"] = "medium"

n = 10000  # horizon for synthetic
# n = 5000  # horizon for heart, wine, movielens

step = np.arange(1, n + 1)
sube = (step.size // 10) * np.arange(1, 11) - 1

mpl.rcParams["font.size"] = 7
plt.figure(figsize=(6.8, 2.25))

def linestyle2dashes(style):
  if style == "--":
    return (3, 3)
  elif style == ":":
    return (0.5, 2.5)
  else:
    return (None, None)

algs = [
  ("LinUCB", {}, "green", "-", "LinUCB"),
  ("LinTS", {}, "yellowgreen", "-", "LinTS"),
  ("LinGreedy", {}, "blue", "-", r"$\varepsilon$-greedy"),
  ("LinExploreCommit", {}, "cyan", "-", "EtC"),
  ("LinPhasedElim", {}, "orange", "-", "Elimination"),
  ("CODE", {}, "red", "-", "CODE")]

# # load data
with open('data_linear_new.pickle', 'rb') as f: # Uncomment to use the new data
# with open('data_movielens.pickle', 'rb') as f:
# with open('data_heart3.pickle', 'rb') as f:
# with open('data_wine.pickle', 'rb') as f:
# with open('data_linear_changing2.pickle', 'rb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    data = pickle.load(f)

regret_algs, simple_regret_algs = data[0], data[1]

# # simulation

for alg in algs:
  
  # all runs for a single algorithm
  # alg_class = globals()[alg[0]]
  # regret, ce = evaluate(alg_class, alg[1], envs, n)
  regret, simple_regret = regret_algs[:, :, algs.index(alg)], simple_regret_algs[:, :, algs.index(alg)]

  # regret plot
  plt.subplot(1, 2, 1)
  cum_regret = regret.cumsum(axis=0)
  plt.plot(step, cum_regret.mean(axis=1),
    alg[2], dashes=linestyle2dashes(alg[3]), label=alg[4])
  plt.errorbar(step[sube], cum_regret[sube, :].mean(axis=1),
    cum_regret[sube, :].std(axis=1) / np.sqrt(cum_regret.shape[1]),
    fmt="none", ecolor=alg[2])

  # simple plot
  plt.subplot(1, 2, 2)
  cum_simple_regret = simple_regret.cumsum(axis=0)
  plt.plot(step, cum_simple_regret.mean(axis=1),
    alg[2], dashes=linestyle2dashes(alg[3]))
  plt.errorbar(step[sube], cum_simple_regret[sube, :].mean(axis=1),
    cum_simple_regret[sube, :].std(axis=1) / np.sqrt(cum_simple_regret.shape[1]),
    fmt="none", ecolor=alg[2])



plt.subplot(1, 2, 1)
plt.title("(a) Linear Bandits",  fontsize=11)
# plt.title("(a) Linear Bandits (Changing Arms)",  fontsize=11)
# plt.title("(a) Movielens",  fontsize=11)
# plt.title("(a) Heart Failure", fontsize=11)
# plt.title("(a) Wine Quality", fontsize=11)
plt.xlabel("Round n", fontsize=9)
plt.ylabel("Regret", fontsize=9)

plt.legend(loc="upper left", ncol=2, frameon=False, fontsize=9)
# plt.yscale('log')
plt.ylim(1, 5000) # required in wine

plt.subplot(1, 2, 2)
plt.title("(b) Linear Bandits",  fontsize=11)
# plt.title("(b) Linear Bandits (Changing Arms)",  fontsize=11)
# plt.title("(b) Movielens",  fontsize=11)
# plt.title("(b) Heart Failure",  fontsize=11)
# plt.title("(b) Wine Quality", fontsize=11)
plt.xlabel("Round n", fontsize=9)
plt.ylabel("Model error", fontsize=9)
# plt.ylim(0.1, 500)
plt.yscale('log')

plt.tight_layout()

plt.savefig("Figures/Linear2_new.pdf", format="pdf", dpi=1200, bbox_inches=0)
plt.savefig("Figures/Linear2_new.png", format="png", dpi=1200, bbox_inches=0)

# plt.savefig("Figures/Linear2_changing.pdf", format="pdf", dpi=1200, bbox_inches=0)
# plt.savefig("Figures/Linear2_changing.png", format="png", dpi=1200, bbox_inches=0)

# plt.savefig("Figures/Linear2_movielens.pdf", format="pdf", dpi=1200, bbox_inches=0)
# plt.savefig("Figures/Linear2_movielens.png", format="png", dpi=1200, bbox_inches=0)

# plt.savefig("Figures/Linear2_heart.pdf", format="pdf", dpi=1200, bbox_inches=0)
# plt.savefig("Figures/Linear2_heart.png", format="png", dpi=1200, bbox_inches=0)

# plt.savefig("Figures/Linear2_wine.pdf", format="pdf", dpi=1200, bbox_inches=0)
# plt.savefig("Figures/Linear2_wine.png", format="png", dpi=1200, bbox_inches=0)

plt.show()