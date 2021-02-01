import numpy as np
from scipy.sparse import csr_matrix


def build_A(ps, ns, alphas):
    n = np.sum(ns)
    cumsum = np.cumsum(ns)
    K = len(ps)
    alpha = ps.dot(alphas)
    Id = np.identity(n)
    zeros = np.zeros((n, K))
    L = np.zeros((n, K))
    for i, g in enumerate(cumsum):
        if i == 0:
            L[:g, 0] -= np.ones(g)
        else:
            L[cumsum[i-1]:g, i] -= np.ones(ns[i])
    G = np.ones((n, K))
    for i, g in enumerate(cumsum):
        if i == 0:
            G[:g, :] *= ps[i] / alpha
            G[:g, :] += L[:g, :] / alphas[i]
        else:
            G[cumsum[i-1]:g, :] *= ps[i] / alpha
            G[cumsum[i-1]:g, :] += L[cumsum[i-1]:g, :] / alphas[i]
    A = np.block([[-Id, L, zeros],[-Id, L, G]])
    return csr_matrix(A)

def build_c(ns, alphas):
    c1 = np.repeat(1 / ns, ns)
    c2 = np.zeros(len(alphas))
    return np.hstack([c1, alphas, c2])

def build_b(ps, alphas, ns, pred_prob=None):
    coef = ps / ps.dot(alphas)
    coef_rep = np.repeat(coef, ns)
    return np.hstack([coef_rep * pred_prob, coef_rep * (1 - pred_prob)])
    pass

def build_l(ns):
    n = np.sum(ns)
    K = len(ns)
    return np.array(([0] * n) + ([None] * (2 * K)))

def build_u(ns):
    n = np.sum(ns)
    K = len(ns)
    return np.array([None] * (n + 2 * K))