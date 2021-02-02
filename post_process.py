import numpy as np
from lp_solver import solve_lp
from lp_transformer import build_params, convert_lp_result
from sklearn.exceptions import NotFittedError
from lp_solver import solve_lp


def predict(s, prob, thresholds, rejects):
    if np.abs(rejects[s][0] - prob) <= rejects[s][1]:
        return 10000.0
    elif prob > thresholds[s]:
        return 1.
    else:
        return 0.


class TransformDPAbstantion():
    def __init__(self, base_classifier, alphas=None):
        self.base = base_classifier
        self.alphas = alphas

    def fit(self, X_unlab):
        sensitives = np.unique(X_unlab[:, -1])
        n = len(X_unlab[:, -1])
        K = len(sensitives)
        assert set(sensitives) == set(self.alphas.keys()), 'Sensitive attributes do not match'
        try:
            prob = self.base.predict_proba(X_unlab)
        except NotFittedError as e:
            print(repr(e))
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
        y_pred = []
        n, _ = X.shape
        # print(self.check_reject_)
        probs = self.base.predict_proba(X)[:, 1]
        # print(probs)
        # print(self.check_reject_[0])
        # print(self.thresholds_)
        for i in range(n):
            s = X[i, -1]
            y_pred.append(predict(s, probs[i], self.thresholds_,
                                  self.check_reject_))
        return np.array(y_pred)