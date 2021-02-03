import numpy as np
from get_data import get_adult
from make_experiment import split_data, run_experiment


RUN_ADULT = True
RUN_MARKET = False
RUN_COMPAS = False
RUN_GERMAN = False
METHODS = ['LR']



setup = {
    "proc_train" : .6,
    "proc_unlab" : .2,
    "cv" : 5,
    "num_c" : 30,
    "num_gamma" : 30,
    "verbose" : 0,
    "n_jobs" : 4
}

check_run = {
    'Adult': RUN_ADULT,
    'Marketing': RUN_MARKET,
    'Compas': RUN_COMPAS,
    'German': RUN_GERMAN
}
datasets = {
            'Compas': None,
            'Marketing': None,
            'German': None,
            'Adult': get_adult
            }
seeds = np.arange(10)
alphas = {
    0: .8,
    1: .8
}
total = len(seeds)

for method in METHODS:
    print('[{}] is running'.format(method))
    for data in datasets.keys():
        if check_run[data]:
            for i, seed in enumerate(seeds):
                print('[{}]: {}/{}'.format(data, i + 1, total))
                X, y = datasets[data]()
                run_experiment(X=X, y=y, alphas=alphas,
                               seed=seed, method=method,
                               **setup)
        else:
            print('[{}] skipped'.format(data))