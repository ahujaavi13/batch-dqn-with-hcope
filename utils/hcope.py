import math

import numpy as np
from scipy.stats import t


def PDIS(theta, X):
    """Per-decision Importance Sampling"""
    num_episodes = X.shape[0]
    est = np.zeros(num_episodes, dtype=float)
    for i in range(num_episodes):
        ep_data = X[i]
        pi = 1
        gamma = 1
        for _t in range(len(ep_data)):
            s, a, r, pi_b = ep_data[_t]
            pi *= theta[int(s)][int(a)] / pi_b
            est[i] += gamma * pi * r
            gamma *= 0.95

    return est


def safetyTest(cS, safetyData_X, delta, c):
    """Check if the safety constraints are satisfied"""
    cS = softmax(cS.reshape(18, 4), axis=1)
    g_samples = PDIS(cS, safetyData_X)
    _lowerBound = calcLowerBound(g_samples, delta)
    if _lowerBound < c:
        return _lowerBound, False
    return _lowerBound, True


def tinv(p, nu):
    return t.ppf(p, nu)


def standard_deviation(v):
    n = v.size
    variance = (np.var(v) * n) / (n - 1)
    return np.sqrt(variance)


def calcLowerBound(v, delta):
    n = v.size
    res = v.mean() - standard_deviation(v) / math.sqrt(n) * tinv(1.0 - delta, n - 1)
    return res


def calcPredictLowerBound(v, delta, k):
    res = v.mean() - 2.0 * standard_deviation(v) / math.sqrt(k) * tinv(1.0 - delta, k - 1)
    return res


def softmax(x, axis):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x, axis=axis)[:, np.newaxis])
    return e_x / e_x.sum(axis=axis)[:, np.newaxis]
