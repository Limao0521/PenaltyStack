"""Bounded random search and (mu/mu_w, lambda)-CMA-ES; both maximise.

`objective(x) -> float` receives a vector inside `bounds` (list of (low,
high)). CMA-ES runs in the unit cube and clips to it, so no candidate ever
leaves the hard bounds.
"""
import math

import numpy as np


def _check(bounds):
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or not (bounds[:, 0] < bounds[:, 1]).all():
        raise ValueError('bounds debe ser una lista de pares (low < high)')
    return bounds


def _evaluate(objective, x, history):
    value = float(objective(x))
    if not math.isfinite(value):
        value = -math.inf
    history.append({'x': x.tolist(), 'value': value})
    return value


def _result(history):
    best = max(history, key=lambda item: item['value'])
    return {'best_x': best['x'], 'best_value': best['value'], 'evaluations': len(history),
            'history': history}


def random_search(objective, bounds, budget, seed=0):
    bounds = _check(bounds)
    rng = np.random.default_rng(seed)
    history = []
    for _ in range(budget):
        _evaluate(objective, rng.uniform(bounds[:, 0], bounds[:, 1]), history)
    return _result(history)


def cma_es(objective, bounds, budget, seed=0, x0=None, sigma0=.3, population=None):
    bounds = _check(bounds)
    n = len(bounds)
    low, span = bounds[:, 0], bounds[:, 1] - bounds[:, 0]
    rng = np.random.default_rng(seed)
    lam = population or 4 + int(3*math.log(n))
    mu = lam // 2
    weights = np.log(mu + .5) - np.log(np.arange(1, mu+1))
    weights /= weights.sum()
    mueff = 1/np.sum(weights**2)
    cc = (4 + mueff/n)/(n + 4 + 2*mueff/n)
    cs = (mueff + 2)/(n + mueff + 5)
    c1 = 2/((n + 1.3)**2 + mueff)
    cmu = min(1 - c1, 2*(mueff - 2 + 1/mueff)/((n + 2)**2 + mueff))
    damps = 1 + 2*max(0, math.sqrt((mueff - 1)/(n + 1)) - 1) + cs
    chi = math.sqrt(n)*(1 - 1/(4*n) + 1/(21*n*n))
    mean = np.full(n, .5) if x0 is None else np.clip((np.asarray(x0, float) - low)/span, 0, 1)
    sigma, cov = sigma0, np.eye(n)
    pc, ps = np.zeros(n), np.zeros(n)
    history = []
    generation = 0
    while len(history) < budget:
        eigval, eigvec = np.linalg.eigh(cov)
        eigval = np.maximum(eigval, 1e-20)
        transform = eigvec @ np.diag(np.sqrt(eigval))
        steps, values = [], []
        for _ in range(min(lam, budget - len(history))):
            u = np.clip(mean + sigma*(transform @ rng.standard_normal(n)), 0, 1)
            steps.append((u - mean)/sigma)
            values.append(_evaluate(objective, low + u*span, history))
        if len(steps) < mu:
            break
        order = np.argsort(values)[::-1][:mu]
        ys = np.array([steps[i] for i in order])
        old = mean
        mean = np.clip(mean + sigma*(weights @ ys), 0, 1)
        step = (mean - old)/sigma
        inv_sqrt = eigvec @ np.diag(1/np.sqrt(eigval)) @ eigvec.T
        ps = (1 - cs)*ps + math.sqrt(cs*(2 - cs)*mueff)*(inv_sqrt @ step)
        generation += 1
        hsig = (np.linalg.norm(ps)/math.sqrt(1 - (1 - cs)**(2*generation))/chi) < 1.4 + 2/(n + 1)
        pc = (1 - cc)*pc + hsig*math.sqrt(cc*(2 - cc)*mueff)*step
        rank_mu = (ys.T*weights) @ ys
        cov = ((1 - c1 - cmu)*cov + c1*(np.outer(pc, pc) + (not hsig)*cc*(2 - cc)*cov)
               + cmu*rank_mu)
        cov = (cov + cov.T)/2
        sigma = min(1., sigma*math.exp((cs/damps)*(np.linalg.norm(ps)/chi - 1)))
    result = _result(history)
    result['final_mean'] = (low + mean*span).tolist()
    return result
