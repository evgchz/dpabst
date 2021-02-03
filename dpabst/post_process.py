import numpy as np
from sklearn.exceptions import NotFittedError
from .lp_transformer import build_params, convert_lp_result
from .lp_solver import solve_lp
from .lp_solver import solve_lp


class TransformDPAbstantion():
    def __init__(self, base_classifier, alphas=None):
        self.base = base_classifier
        self.alphas = alphas

    def fit(self, X_unlab):
        sensitives = np.unique(X_unlab[:, -1])
        if set(self.base.classes_) != set([0., 1.]):
            raise ValueError('Target variable is not valued in 0/1')
        if set(sensitives) != set(self.alphas.keys()):
            raise ValueError('Groups do not match: data {}, alphas {}'.format(set(sensitives), set(self.alphas.keys())))
        n = len(X_unlab[:, -1])
        K = len(sensitives)
        prob = self.base.predict_proba(X_unlab)
        prob = prob[:, 1]
        ps, ns, pred_prob, alphas = build_params(X_unlab, prob, self.alphas)
        res = solve_lp(ps, ns, alphas, pred_prob)
        alpha = ps.dot(alphas)
        lambdas, gammas = convert_lp_result(res, ns, K)
        self.thresholds_ = {}
        self.check_reject_ = {}
        for i, s in enumerate(sensitives):
            a = .5 * (1 - gammas.sum()) + (alpha * gammas[i]) / (2 * alphas[i] * ps[i])
            b = .5 * (1 - gammas.sum()) + (alpha * gammas[i]) / (2 * alphas[i] * ps[i]) + lambdas[i] * alpha / ps[i]
            c = .5 + (alpha * gammas[i] / (alphas[i] * ps[i]) - gammas.sum()) / 2
            self.thresholds_[s] = c
            self.check_reject_[s] = [a, b]
            # print(lambdas)

    def predict(self, X):
        # stupid implementation
        n, _ = X.shape
        probs = self.base.predict_proba(X)[:, 1]

        y_pred = np.zeros(n)
        sensitives = np.unique(X[:, -1])

        for s in sensitives:
            s_mask = X[:, -1] == s
            a = self.check_reject_[s][0]
            b = self.check_reject_[s][1]
            c = self.thresholds_[s]
            m_pos = np.where((X[:, -1] == s) & (probs > c), True, False)
            m_rej = np.where((X[:, -1] == s) & (np.abs(a - probs) <= b),
                             True, False)
            y_pred[m_pos] = 1.
            y_pred[m_rej] = 10000.
        return y_pred