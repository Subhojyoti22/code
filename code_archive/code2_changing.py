# -*- coding: utf-8 -*-
"""CODE2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kV9EF6U_YdWozOl9nA8bOU-GQtpXpQTH
"""

# Imports and defaults
import joblib
from joblib import Parallel, delayed
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import sqrtm
from scipy.optimize import linprog
import time
import pickle

import ipdb

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

import platform
print("python %s" % platform.python_version())
print("matplotlib %s" % mpl.__version__)
print("%d joblib CPUs" % joblib.cpu_count())

def linestyle2dashes(style):
  if style == "--":
    return (3, 3)
  elif style == ":":
    return (0.5, 2.5)
  else:
    return (None, None)

# Optimal designs
def d_grad(X, V, p, return_grad=True):
  """Value of D-optimal objective and its gradient.

  X: n x d matrix of arm features
  V: prior design matrix
  p: distribution over n arms (design)
  """
  n, d = X.shape

  # inverse of the sample covariance matrix
  Xp = X * np.sqrt(p[:, np.newaxis])
  G = V + Xp.T.dot(Xp)
  invG = np.linalg.inv(G)

  # objective value (log det)
  _, obj = np.linalg.slogdet(invG)
  if return_grad:
    # gradient of the objective
    X2 = np.einsum("ki,kj->kij", X, X)
    M = np.einsum("kij,jl->kil", X2, invG)
    dp = - np.trace(M, axis1=-2, axis2=-1)
  else:
    dp = 0

  return obj, dp


def d_design(X, V=None, pi_0=None, num_iters=100, tol=1e-6, printout=True):
  """Frank-Wolfe algorithm for d-design optimization.

  X: n x d matrix of arm features
  V: prior design matrix
  pi_0: initial distribution over n arms (design)
  num_iters: maximum number of Frank-Wolfe iterations
  tol: stop when two consecutive objective values differ by less than tol
  """
  n, d = X.shape

  if V is None:
    V = np.zeros((d, d))
  V += 1e-6 * np.eye(d)  # avoiding singularity

  if pi_0 is None:
    # initial allocation weights are 1 / n and they add up to 1
    pi = np.ones(n) / n
  else:
    pi = np.copy(pi_0)

  # initialize constraints
  A_ub_fw = np.ones((1, n))
  b_ub_fw = 1

  # Frank-Wolfe iterations
  for iter in range(num_iters):
    # compute gradient at the last solution
    pi_last = np.copy(pi)
    last_obj, grad = d_grad(X, V, pi_last)

    if printout:
      print("%.4f" % last_obj, end=" ")

    # find a feasible LP solution in the direction of the gradient
    result = linprog(grad, A_ub_fw, b_ub_fw, bounds=[0, 1], method="highs")
    pi_lp = result.x
    pi_lp = np.maximum(pi_lp, 0)
    pi_lp /= pi_lp.sum()

    # line search in the direction of the gradient
    w = np.append(np.logspace(-10, 0, 21, base=2), 0)
    pi_ = np.outer(w, pi_lp) + np.outer(1 - w, pi_last)
    G = V[np.newaxis, :, :] + np.einsum("pi,ij,ik->pjk", pi_, X, X)
    _, obj = np.linalg.slogdet(G)
    best = np.argmax(obj)

    # update solution
    pi = w[best] * pi_lp + (1 - w[best]) * pi_last

    if np.abs(pi - pi_last).sum() < tol:
      break;
    iter += 1

  if printout:
    print()

  pi = np.maximum(pi, 0)
  pi /= pi.sum()
  return pi

# Bandit environments and simulator
class LinBandit(object):
  """Linear bandit."""

  def __init__(self, X, theta, sigma=0.5):
    self.X = np.copy(X)  # K x d matrix of arm features
    self.K = self.X.shape[0]
    self.d = self.X.shape[1]
    self.theta = np.copy(theta)  # model parameter
    self.sigma = sigma  # reward noise

    self.mu = self.X.dot(self.theta)  # mean rewards of all arms
    self.best_arm = np.argmax(self.mu)  # optimal arm

    self.randomize()

  def randomize(self):
    # generate random rewards
    self.rt = self.mu + self.sigma * np.random.randn(self.K)

  def reward(self, arm):
    # instantaneous reward of the arm
    return self.rt[arm]

  def regret(self, arm):
    # instantaneous regret of the arm
    return self.rt[self.best_arm] - self.rt[arm]

  def pregret(self, arm):
    # expected regret of the arm
    return self.mu[self.best_arm] - self.mu[arm]

  def print(self):
    return "Linear bandit: %d dimensions, %d arms" % (self.d, self.K)

def evaluate_one(Alg, params, env_total, n):
  """One run of a bandit algorithm."""
  
  env = env_total[0]
  alg = Alg(env, n, params)

  regret = np.zeros(n)
  pulled_arms = np.zeros(n, dtype=int)
  errors = np.zeros((n, env.K))
  for t in range(n):
    # generate state
    env = env_total[t]
    alg.env = env
    env.randomize()

    # take action and update agent
    arm = alg.get_arm(t)
    alg.update(t, arm, env.reward(arm))

    # track performance
    regret_at_t = env.regret(arm)
    regret[t] += regret_at_t
    pulled_arms[t] = arm
    errors[t, :] = np.square(env.X.dot(alg.get_mle() - env.theta))
    # errors[t, :] = env.X.dot(alg.get_mle())

  metric = np.zeros(n)
  future_pulls = np.zeros(env.K, dtype=int)
  for t in range(n - 1, -1, -1):
    future_pulls[pulled_arms[t]] += 1
    metric[t] = errors[t, future_pulls > 0].max()
    # metric[t] = (np.argmax(errors[t, future_pulls > 0]) != np.argmax(env.mu[future_pulls > 0]))

  return regret, metric


def evaluate(Alg, params, env, n=1000, printout=True):
  """Multiple runs of a bandit algorithm."""
  if printout:
    print("Evaluating %s" % Alg.print(), end="")
  start = time.time()

  num_exps = len(env)
  regret = np.zeros((n, num_exps))
  metric = np.zeros((n, num_exps))

  output = Parallel(n_jobs=1)(delayed(evaluate_one)(Alg, params, env[ex, :], n)
    for ex in range(num_exps))
  for ex in range(num_exps):
    regret[:, ex] = output[ex][0]
    metric[:, ex] = output[ex][1]
  if printout:
    print(" %.1f seconds" % (time.time() - start))

  if printout:
    total_regret = regret.sum(axis=0)
    total_simple_regret = metric.sum(axis=0)

    print("Regret: %.2f +/- %.2f (median: %.2f, max: %.2f, min: %.2f)" %
      (total_regret.mean(), total_regret.std() / np.sqrt(num_exps),
      np.median(total_regret), total_regret.max(), total_regret.min()))
    
    print("Simple regret: %.2f +/- %.2f (median: %.2f, max: %.2f, min: %.2f)" %
      (total_simple_regret.mean(), total_simple_regret.std() / np.sqrt(num_exps),
      np.median(total_simple_regret), total_simple_regret.max(), total_simple_regret.min()))

  return regret, metric

# Bandit algorithms
class LinBanditAlg:
  def __init__(self, env, n, params):
    self.env = env  # bandit environment that the agent interacts with
    self.K = self.env.K  # number of arms
    self.d = self.env.d  # number of features
    self.n = n  # horizon
    self.theta0 = np.zeros(self.d)  # prior mean of the model parameter
    self.Sigma0 = np.eye(self.d)  # prior covariance of the model parameter
    self.sigma = 0.5  # reward noise

    # override default values
    for attr, val in params.items():
      if isinstance(val, np.ndarray):
        setattr(self, attr, np.copy(val))
      else:
        setattr(self, attr, val)

    # sufficient statistics
    self.Lambda = np.linalg.inv(self.Sigma0)
    self.B = self.Lambda.dot(self.theta0)

  def update(self, t, arm, r):
    # update sufficient statistics
    x = self.env.X[arm, :]
    self.Lambda += np.outer(x, x) / np.square(self.sigma)
    self.B += x * r / np.square(self.sigma)

  def get_mle(self):
    thetahat = np.linalg.solve(self.Lambda, self.B)
    return thetahat

class LinTS(LinBanditAlg):
  def get_arm(self, t):
    # linear model posterior
    Sigmahat = np.linalg.inv(self.Lambda)
    thetahat = Sigmahat.dot(self.B)

    # posterior sampling
    thetatilde = np.random.multivariate_normal(thetahat, Sigmahat)
    self.mu = self.env.X.dot(thetatilde)

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "LinTS"


class LinUCB(LinBanditAlg):
  def __init__(self, env, n, params):
    LinBanditAlg.__init__(self, env, n, params)

    self.cew = self.confidence_ellipsoid_width(n)

  def confidence_ellipsoid_width(self, t):
    # Theorem 2 in Abassi-Yadkori (2011)
    # Improved Algorithms for Linear Stochastic Bandits
    delta = 1 / self.n
    L = np.amax(np.linalg.norm(self.env.X, axis=1))
    Lambda = np.square(self.sigma) * np.linalg.eigvalsh(np.linalg.inv(self.Sigma0)).max()  # V = \sigma^2 (posterior covariance)^{-1}
    R = self.sigma
    S = np.sqrt(self.d)
    width = np.sqrt(Lambda) * S + \
      R * np.sqrt(self.d * np.log((1 + t * np.square(L) / Lambda) / delta))
    return width

  def get_arm(self, t):
    # linear model posterior
    Sigmahat = np.linalg.inv(self.Lambda)
    thetahat = Sigmahat.dot(self.B)

    # UCBs
    invV = Sigmahat / np.square(self.sigma)  # V^{-1} = posterior covariance / \sigma^2
    self.mu = self.env.X.dot(thetahat) + self.cew * \
      np.sqrt((self.env.X.dot(invV) * self.env.X).sum(axis=1))

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "LinUCB"


class LinGreedy(LinBanditAlg):
  def __init__(self, env, n, params):
    self.epsilon = 0.05

    LinBanditAlg.__init__(self, env, n, params)

  def get_arm(self, t):
    self.mu = np.zeros(self.K)

    if np.random.rand() < self.epsilon * np.sqrt(self.n / (t + 1)) / 2:
      self.mu[np.random.randint(self.K)] = np.Inf
    else:
      theta = np.linalg.solve(self.Lambda, self.B)
      self.mu = self.env.X.dot(theta)

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "Linear e-greedy"


class LinExploreCommit(LinBanditAlg):
  def __init__(self, env, n, params):
    self.epsilon = 0.05

    LinBanditAlg.__init__(self, env, n, params)

  def get_arm(self, t):
    self.mu = np.zeros(self.K)

    if t <= np.round(self.epsilon * self.n):
      self.mu[np.random.randint(self.K)] = np.Inf
      if t == np.round(self.epsilon * self.n):
        self.theta = np.linalg.solve(self.Lambda, self.B)
    else:
      self.mu = self.env.X.dot(self.theta)

    arm = np.argmax(self.mu)
    return arm

  @staticmethod
  def print():
    return "Linear explore-then-commit"


class LinPhasedElim(LinBanditAlg):
  def __init__(self, env, n, params):
    self.delta = 0.05  # confidence interval failure probability
    self.reset_statistics = True

    LinBanditAlg.__init__(self, env, n, params)

    # Section 22 in "Bandit Algorithms"
    self.phase = 0
    ell = self.phase + 1
    self.remaining_rounds = int(np.ceil(2 * d * np.power(4, ell) * \
      np.log(self.K * ell * (ell + 1) / self.delta)))
    self.active_arms = np.arange(self.K)

    # optimal design
    self.pi = d_design(self.env.X, num_iters=100, printout=False)

  def get_arm(self, t):
    if not self.remaining_rounds:
      # linear model posterior
      Sigmahat = np.linalg.inv(self.Lambda)
      thetahat = Sigmahat.dot(self.B)

      # elimination
      ci = np.power(0.5, self.phase + 1)
      UCB = self.env.X.dot(thetahat) + ci
      LCB = self.env.X.dot(thetahat) - ci
      self.active_arms = self.active_arms[UCB[self.active_arms] > LCB[self.active_arms].max()]

      # initialize a new phase
      self.phase += 1
      ell = self.phase + 1
      self.remaining_rounds = int(np.ceil(2 * d * np.power(4, ell) * \
        np.log(self.K * ell * (ell + 1) / self.delta)))

      if self.reset_statistics:
        # sufficient statistics
        self.Lambda = np.linalg.inv(self.Sigma0)
        self.B = self.Lambda.dot(self.theta0)

      # optimal design
      if self.active_arms.size > 1:
        self.pi = d_design(self.env.X[self.active_arms, :], num_iters=100, printout=False)
        self.pi /= self.pi.sum()
      else:
        self.pi = np.ones(1)
    else:
      self.remaining_rounds -= 1

    arm = np.random.choice(self.active_arms, p=self.pi)
    return arm

  @staticmethod
  def print():
    return "Linear phased elimination"


class CODE(LinBanditAlg):
  def __init__(self, env, n, params):
    self.acquisition = "policy"
    self.delta = 0.05  # confidence interval failure probability

    LinBanditAlg.__init__(self, env, n, params)
    self.L = 1
    self.prior_effect = 1e5 * np.square(self.sigma)  # V = \sigma^2 (posterior covariance)^{-1}

  def confidence_ellipsoid_width(self, t):
    # Theorem 2 in Abassi-Yadkori (2011)
    # Improved Algorithms for Linear Stochastic Bandits
    R = self.sigma
    S = 0
    width = np.sqrt(self.prior_effect) * S + \
      R * np.sqrt(self.d * np.log((1 + t * np.square(self.L) / self.prior_effect) / self.delta))
    return width

  def get_arm(self, t):
    # linear model posterior
    Sigmahat = np.linalg.inv(self.Lambda)
    thetahat = Sigmahat.dot(self.B)

    # elimination
    cew = self.confidence_ellipsoid_width(t)
    invV = Sigmahat / np.square(self.sigma)  # V^{-1} = posterior covariance / \sigma^2
    ci = cew * np.sqrt((self.env.X.dot(invV) * self.env.X).sum(axis=1))
    UCB = self.env.X.dot(thetahat) + ci
    LCB = self.env.X.dot(thetahat) - ci
    self.active_arms = np.flatnonzero(UCB > LCB.max())

    if self.acquisition == "action":
      # maximum variance action
      X_active = self.env.X[self.active_arms, :]
      var = np.einsum("ij,jk,ik->i", X_active, Sigmahat, X_active)
      best = np.argmax(var)
      arm = self.active_arms[best]
    elif self.acquisition == "policy":
      if self.active_arms.size > 1:
        # minimum D-optimal design policy
        num_iters = 2 * self.d
        pi = d_design(self.env.X[self.active_arms, :], self.Lambda, num_iters=num_iters, tol=1e-4, printout=False)
        best = np.random.choice(self.active_arms.size, p=pi)
        arm = self.active_arms[best]
      else:
        arm = self.active_arms[0]
    else:
      raise Exception("Unknown acquisition function in LinOD.")

    return arm

  @staticmethod
  def print():
    return "CODE"

def generate_bandits(num_runs, n, theta0, Sigma0):
  # envs = []
  envs = np.zeros((num_runs, n), dtype=object)
  for run in range(num_runs):
    # sample model parameter
    theta = np.random.multivariate_normal(theta0, Sigma0)
    # sample arm features from a hypercubecube
    X = 2 * np.random.rand(5*K, d) - 1
    # ipdb.set_trace()
    for i in range(n):
        choose_subset_X_indx = np.random.choice(5*K, K, replace=False)
        new_X = X[choose_subset_X_indx, :]
        # initialize bandit environment
        # envs.append(LinBandit(X, theta))
        envs[run, i] = LinBandit(new_X, theta)

  return envs


d = 8  # number of features
K = 200  # number of arms
n = 10000  # horizon
num_runs = 50  # number of random runs

algs = [
  ("LinUCB", {}, "green", "-", "LinUCB"),
  ("LinTS", {}, "yellowgreen", "-", "LinTS"),
  ("LinGreedy", {}, "blue", "-", r"$\varepsilon$-greedy"),
  ("LinExploreCommit", {}, "cyan", "-", "EtC"),
  ("LinPhasedElim", {}, "orange", "-", "Elimination"),
  ("CODE", {"acquisition": "action"}, "red", "-", "CODE")]

# prior distribution of the model parameter
theta0 = np.zeros(d)  # prior mean
Sigma0 = np.eye(d)  # prior covariance (this is standard deviation in the MAB)

# bandit environments
envs = generate_bandits(num_runs, n, theta0, Sigma0)

step = np.arange(1, n + 1)  # for plots
sube = (step.size // 10) * np.arange(1, 11) - 1

mpl.rcParams["font.size"] = 7
plt.figure(figsize=(6.8, 2.25))

regret_algs = np.zeros((n, num_runs, len(algs)))
simple_regret_algs = np.zeros((n, num_runs, len(algs)))

# simulation
for alg in algs:
  # all runs for a single algorithm
  alg_class = globals()[alg[0]]
  regret, simple_regret = evaluate(alg_class, alg[1], envs, n)

  regret_algs[:, :, algs.index(alg)] = regret
  simple_regret_algs[:, :, algs.index(alg)] = simple_regret

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


with open('data_linear_changing2.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
    pickle.dump([regret_algs, simple_regret_algs], f)
f.close()



plt.subplot(1, 2, 1)
plt.title("(a) Linear bandit")
plt.xlabel("Round n")
plt.ylabel("Regret")
plt.ylim(0, 3000)
plt.legend(loc="upper left", ncol=2, frameon=False)

plt.subplot(1, 2, 2)
plt.title("(b) Linear bandit")
plt.xlabel("Round n")
plt.ylabel("Model error")
plt.ylim(0.1, 500)

plt.tight_layout()
plt.savefig("Figures/Linear3_changing.pdf", format="pdf", dpi=1200, bbox_inches=0)
plt.savefig("Figures/Linear3_changing.png", format="png", dpi=1200, bbox_inches=0)
plt.show()
plt.show()