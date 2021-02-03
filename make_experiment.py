import numpy as np
from post_process import TransformDPAbstantion
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV


def risk(y_true, y_pred, x_sens):

    sensitives = np.unique(x_sens)
    report_accuracy = {}
    for s in sensitives:
        mask_s = (x_sens == s)
        mask = (x_sens == s) & (y_pred != 10000.)
        pr = mask.sum() / mask_s.sum()
        report_accuracy['clf_rate_'+str(s)] = pr

    corect = y_true[(y_pred != 10000.)] == y_pred[(y_pred != 10000.)]
    accuracy_overall = np.mean(corect)
    report_accuracy['acc_total'] = accuracy_overall

    # We split it to have better order of keys ... (Stupid, we know)
    for s in sensitives:
        mask = (x_sens == s) & (y_pred != 10000.)
        acc_s = np.mean((y_true[mask] == y_pred[mask]))
        report_accuracy['acc_'+str(s)] = acc_s
    return report_accuracy


def compute_dp(y_pred, x_sens):
    sensitives = np.unique(x_sens)
    report_fairness = {}
    for s in sensitives:
        mask = (x_sens == s) & (y_pred != 10000.)
        pos_r = y_pred[mask].sum() / mask.sum()
        report_fairness['pos_rate_{}'.format(str(s))] = pos_r
    return report_fairness


def print_report(report):
    for key in report.keys():
        print('[REPORT]: {} is {:.3f}'.format(key, report[key]))



def split_data(X, y, proc_train, proc_unlab, seed=None, shuffle=True):
    n = len(y)
    n_train = int(proc_train * n)
    n_unlab = int(proc_unlab * n)
    np.random.seed(seed)
    if shuffle:
        permute = np.random.permutation(len(y))
        y = y[permute]
        X = X[permute]
    X_train, y_train = X[:n_train, :], y[:n_train]
    X_unlab = X[n_train:n_train + n_unlab, :]
    X_test, y_test = X[n_train + n_unlab:, :], y[n_train + n_unlab:]
    return X_train, y_train, X_unlab, X_test, y_test


def run_experiment(X, y, alphas, seed, proc_train,
                   proc_unlab, cv, num_c, num_gamma,
                   verbose, n_jobs):
    X_train, y_train, X_unlab, X_test, y_test = split_data(X, y, proc_train,
                                                           proc_unlab, seed)
    scaler = StandardScaler()
    scaler.fit(X_train[:, :-1])

    X_train[:, :-1] = scaler.transform(X_train[:, :-1])
    X_unlab[:, :-1] = scaler.transform(X_unlab[:, :-1])
    X_test[:, :-1] = scaler.transform(X_test[:, :-1])
    n_train, d = X_train.shape
    if n_train > d:
        dual = False
    else:
        dual = True

    methods = {
        "LR" : LogisticRegression(solver='liblinear'),
        "L-SVC": CalibratedClassifierCV(LinearSVC(dual=dual)),
        "RF" : RandomForestClassifier()
        # "RBF-SVC": SVC(probability=True),
    }


    Cs = np.logspace(-4, 4, num_c)
    gammas = np.logspace(-4, 4, num_gamma)
    pows = np.array([1, 15/16, 7/8, 3/4, 1/2, 1/4, 1/8, 1/16, 0])
    ds = np.unique((d ** pows).astype('int'))

    parameters = {
        "LR" : {"C" : Cs},
        "L-SVC" : {"base_estimator__C" : Cs},
        "RF" : {"max_features" : ds}
        # "RBF-SVC" : {"C" : Cs, "gamma" : gammas}
    }
    for key in methods.keys():
        clf = GridSearchCV(methods[key], parameters[key],
                           cv=cv, refit=True, verbose=verbose,
                           n_jobs=n_jobs)
        clf.fit(X_train, y_train)
        transformer = TransformDPAbstantion(clf, alphas)
        transformer.fit(X_unlab)
        y_pred = transformer.predict(X_test)
        print('[RESULTS]: {} summary'.format(key))
        print('[Statistics: test data]')
        report_fairness = compute_dp(y_test, X_test[:, -1])
        print_report(report_fairness)

        print('[Statistics: base algorithm]')
        y_pred_unf = clf.predict(X_test)
        report_accuracy0 = risk(y_test, y_pred_unf, X_test[:, -1])
        report_fairness0 = compute_dp(y_pred_unf, X_test[:, -1])
        print_report(report_accuracy0)
        print_report(report_fairness0)

        print('[Statistics: our algorithm]')
        report_accuracy1 = risk(y_test, y_pred, X_test[:, -1])
        report_fairness1 = compute_dp(y_pred, X_test[:, -1])
        print_report(report_accuracy1)
        print_report(report_fairness1)
    # break