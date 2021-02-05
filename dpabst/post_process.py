import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator
from .lp_transformer import build_params, convert_lp_result
from .lp_solver import solve_lp
from .lp_solver import solve_lp


class TransformDPAbstantion(BaseEstimator):
    def __init__(self, base_classifier, alphas,
                 randomize=False, noise=1e-3):
        """Transform base classifier into a classifier with abstention Demographic Parity constraints

        Parameters
        ----------
        base_classifier : estimator object
            Estimator needs to provide ``predict_proba`` function.
            WARNING: it is assumed that the base classifier is already fitted and that the labels are ZERO and ONE. An Error will be raised if it is not the case during fit stage.
        alphas : dict
            Dictionary with keys being sensitive attributes and values being classification rates per sensitive attribute. Keys assumed to be numeric, coinciding with the last column of the design matrix during fit and predict.
        randomize : bool
            If True, then uniformly distribute noise on [0, noise] is added to the probabilities predicted by base_classifier
        noise : float,
            Used only if randomize=True.
        """
        self.base = base_classifier
        self.alphas = alphas
        self.randomize = randomize
        self.noise = noise

    def fit(self, X_unlab):
        """Fits the method

        Parameters
        ----------
        X_unlab : array-like
            It is assumed that the sensitive attribute is stored in the LAST column. An Error will be raised if it is not the case.
        """
        sensitives = np.unique(X_unlab[:, -1])
        if set(self.base.classes_) != set([0., 1.]):
            raise ValueError('Target variable is not valued in 0/1')
        if set(sensitives) != set(self.alphas.keys()):
            raise ValueError('Groups do not match: data {}, alphas {}'.format(set(sensitives), set(self.alphas.keys())))
        n = len(X_unlab[:, -1])
        K = len(sensitives)
        prob = self.base.predict_proba(X_unlab)
        prob = prob[:, 1]
        if self.randomize:
            prob += np.random.uniform(0, self.noise, n)
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

    def predict(self, X):
        """Predict

        Parameters
        ----------
        X : array-like
            It is assumed that the sensitive attribute is stored in the LAST column. An Error will be raised if it is not the case.

        """
        sensitives = np.unique(X[:, -1])
        if set(sensitives) != set(self.alphas.keys()):
            raise ValueError('Groups do not match: data {}, alphas {}'.format(set(sensitives), set(self.alphas.keys())))
        n, _ = X.shape
        probs = self.base.predict_proba(X)[:, 1]
        if self.randomize:
            probs += np.random.uniform(0, self.noise, n)

        y_pred = np.zeros(n)
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