import numpy as np
from get_data import get_adult, get_german, get_compas
from make_experiment import split_data, run_experiment
import pickle
import os


def save_obj(obj, name):
    with open('results/'+ name + '.pkl', 'wb+') as f:
        pickle.dump(obj, f, 0)

def load_obj(name):
    with open('results/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)



RUN_ADULT = False
RUN_MARKET = False
RUN_COMPAS = False
RUN_GERMAN = True
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
            'Compas': get_compas,
            'Marketing': None,
            'German': get_german,
            'Adult': get_adult
            }
seeds = np.arange(20)

# setting alphas to a number means that all groups have the same reject rate
alphas_grid = np.linspace(.5, .95, 10)

total = len(seeds)

if not os.path.exists('results'):
    os.makedirs('results')
if not os.path.exists('results/models'):
    os.makedirs('results/models')
for method in METHODS:
    print('[{}] is running'.format(method))
    for data in datasets.keys():
        if not check_run[data]:
                print('[{}] skipped'.format(data))
        else:
            results = {}
            X, y = datasets[data]()
            sensitives = np.unique(X[:, -1])
            for alphas in alphas_grid:
                print('[Classification rate] {}'.format(alphas))
                for i, seed in enumerate(seeds):
                    print('[{}]: {}/{}'.format(data, i + 1, total))
                    result_seed = run_experiment(X=X, y=y, alphas=alphas,
                                                 seed=seed, method=method,
                                                 data_name=data, **setup)
                    results[alphas, seed] = result_seed
            SAVE_NAME = '{}_{}'.format(data, method)
            save_obj(results, SAVE_NAME)
