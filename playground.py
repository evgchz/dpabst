import numpy as np
import time
from lp_solver import solve_lp
from lp_transformer import build_ps
from post_process import TransformDPAbstantion
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.utils.validation import check_is_fitted
from fairlearn.datasets import fetch_adult

def risk(y_true, y_pred, x_sens):
    n = len(y_true)
    num = 0
    den = 0
    for i in range(n):
        if y_pred[i] != 10000.:
            den += 1
            if y_pred[i] == y_true[i]:
                num += 1

    sensitives = np.unique(x_sens)
    for s in sensitives:
        mask_s = (x_sens == s)
        mask = (x_sens == s) & (y_pred != 10000.)
        pr = mask.sum() / mask_s.sum()
        print('[Predicted]: classification rate for s = {} is {:.3f}'.format(s, pr))
    return num / den


def compute_dp(y_pred, x_sens, label):
    sensitives = np.unique(x_sens)
    for s in sensitives:
        mask = (x_sens == s) & (y_pred != 10000.)
        pos_r = y_pred[mask].sum() / mask.sum()
        print('[{}]: positive rate for s = {} is {:.3f}'.format(label, s,
                                                                pos_r))



def get_adult(shuffle=True, seed=None):
    np.random.seed(seed)
    adult = fetch_adult()
    X = adult["data"]
    y = adult["target"]

    X[:, [9,-1]] = X[:, [-1,9]] # puts sex into the last column

    # Erase nan values
    y = y[~np.isnan(X).any(axis=1)]
    X = X[~np.isnan(X).any(axis=1)]

    # shuffle data
    if shuffle:
        permute = np.random.permutation(len(y))
        y = y[permute]
        X = X[permute]

    # transform to binary and set proper type
    X = X.astype('float')
    for i, lab in enumerate(y):
        if y[i] == "<=50K":
            y[i] = 0.
        else:
            y[i] = 1.
    y = y.astype('float')
    return X, y


X, y = get_adult(True, 42)

n_train = 20000
n_unlab = 10000
X_train, y_train = X[:n_train, :], y[:n_train]
X_unlab = X[n_train:n_train + n_unlab, :]
X_test, y_test = X[n_train + n_unlab:, :], y[n_train + n_unlab:]



clf = LogisticRegression()
clf.fit(X_train, y_train)


alphas = {
    0: .8,
    1: .8
}


transformer = TransformDPAbstantion(clf, alphas)
transformer.fit(X_unlab)


y_pred = transformer.predict(X_test)
print('[Accuracy]: {:.3f}'.format(risk(y_test, y_pred, X_test[:, -1])))
compute_dp(y_pred, X_test[:, -1], label='Predicted')



compute_dp(y_test, X_test[:, -1], label='True')

# some random init, to test
# K = 4
# # ps = np.random.dirichlet(np.ones(K))
# # ps = np.array([0.5, 0.5])
# ns = np.random.randint(10, 100, K)
# # ns = np.array([1, 1])
# alphas = np.random.uniform(0.5, 1, K)
# # alphas = .9 * np.ones(K)
# pred_prob = np.random.uniform(0, 1, np.sum(ns))

# n = ns.sum()
# ps = build_ps(ns)
# # print('n = {}, K = {}'.format(n, K))
# # print('Starting to solve lp ...')
# # t0 = time.time()
# # res = solve_lp(ps, ns, alphas, pred_prob)
# # t1 = time.time()
# # print('Finished in {:.1f} sec'.format(t1 - t0))
# # print('--- results ---')
# # print(res)