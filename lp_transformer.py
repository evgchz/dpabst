import numpy as np


def sort_pred_prob(pred_prob, X_unlab):
    # sensitive feature is always assumed to be at the last place of the data matrix!
    sensitives = np.unique(X_unlab[:, -1])
    pre_prob_sorted = np.array([])
    for s in sensitives:
        mask = X_unlab[:, - 1] == s
        pre_prob_sorted = np.concatenate([pre_prob_sorted, pred_prob[mask]])
    return np.array(pre_prob_sorted)

def build_ns(X_unlab):
    sensitives = np.unique(X_unlab[:, -1])
    ns = []
    for s in sensitives:
        ns.append((X_unlab[:, -1] == s).sum())
    return np.array(ns)

def build_ps(ns):
    return ns / np.sum(ns)

def set_alphas(alphas_dict, X_unlab):
    sensitives = np.unique(X_unlab[:, -1])
    alphas = []
    for s in sensitives:
        alphas.append(alphas_dict[s])
    return np.array(alphas)

def build_params(X_unlab, prob, alphas_dict):
    ns = build_ns(X_unlab)
    ps = build_ps(ns)
    prob = sort_pred_prob(prob, X_unlab)
    alphas = set_alphas(alphas_dict, X_unlab)
    return ps, ns, prob, alphas

def convert_lp_result(res, ns, K):
    lambdas = res['x'][ns.sum() : ns.sum() + K]
    gammas = res['x'][ns.sum() + K :]
    return lambdas, gammas