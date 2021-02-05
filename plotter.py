import numpy as np
import pickle
import matplotlib.pyplot as plt


def load_obj(name):
    with open('results/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)



def reject_vs_alpha_per_group(results, seeds, alpha_grid, title, sensitives):
    accuracy_our = {}
    acc_mean_our = {}
    acc_var_our = {}
    for s in sensitives:
        acc_mean_our[s] = []
        acc_var_our[s] = []
    for alpha in alpha_grid:
        for s in sensitives:
            accuracy_our[alpha] = {}
            accuracy_our[alpha]['per_seed{}'.format(s)] = []
            for seed in seeds:
                    accuracy_our[alpha]['per_seed{}'.format(s)].append(results[alpha, seed]['our']['rej{}'.format(s)])
            accuracy_our[alpha]['mean{}'.format(s)] = np.mean(accuracy_our[alpha]['per_seed{}'.format(s)])
            accuracy_our[alpha]['var{}'.format(s)] = np.var(accuracy_our[alpha]['per_seed{}'.format(s)])

            acc_mean_our[s].append(accuracy_our[alpha]['mean{}'.format(s)])
            acc_var_our[s].append(accuracy_our[alpha]['var{}'.format(s)])


    plt.figure('Reject')
    plt.title(title)
    for s in sensitives:
        plt.plot(alpha_grid, acc_mean_our[s], label='Our: s={}'.format(s))
        plt.fill_between(alpha_grid,
                         acc_mean_our[s] - np.sqrt(acc_var_our[s]),
                         acc_mean_our[s] + np.sqrt(acc_var_our[s]),
                         alpha=0.3)
    plt.legend()
    plt.xlabel('alpha')
    plt.xlim(0.499, 0.951)
    plt.ylabel('reject')
    plt.show()





def accuracy_vs_alpha(results, seeds, alpha_grid, title):
    accuracy_our = {}
    accuracy_base = {}
    acc_mean_our = []
    acc_var_our = []
    acc_mean_base = []
    acc_var_base = []
    for alpha in alpha_grid:
        accuracy_our[alpha] = {}
        accuracy_our[alpha]['per_seed'] = []
        accuracy_base[alpha] = {}
        accuracy_base[alpha]['per_seed'] = []
        for seed in seeds:
            accuracy_our[alpha]['per_seed'].append(results[alpha, seed]['our']['acc'])
            accuracy_base[alpha]['per_seed'].append(results[alpha, seed]['base']['acc'])
        accuracy_our[alpha]['mean'] = np.mean(accuracy_our[alpha]['per_seed'])
        accuracy_our[alpha]['var'] = np.var(accuracy_our[alpha]['per_seed'])
        acc_mean_our.append(accuracy_our[alpha]['mean'])
        acc_var_our.append(accuracy_our[alpha]['var'])
        accuracy_base[alpha]['mean'] = np.mean(accuracy_base[alpha]['per_seed'])
        accuracy_base[alpha]['var'] = np.var(accuracy_base[alpha]['per_seed'])
        acc_mean_base.append(accuracy_base[alpha]['mean'])
        acc_var_base.append(accuracy_base[alpha]['var'])


    plt.figure('Accuracy Total')
    plt.title(title)
    plt.plot(alpha_grid, acc_mean_our, label='Our')
    plt.fill_between(alpha_grid,
                     acc_mean_our - np.sqrt(acc_var_our),
                     acc_mean_our + np.sqrt(acc_var_our), alpha=0.3)
    plt.plot(alpha_grid, acc_mean_base, label='Base')
    plt.fill_between(alpha_grid,
                     acc_mean_base - np.sqrt(acc_var_base),
                     acc_mean_base + np.sqrt(acc_var_base), alpha=0.3)
    plt.legend()
    plt.xlabel('alpha')
    plt.xlim(0.501, 0.949)
    plt.ylabel('accuracy')



def accuracy_vs_alpha_per_group(results, seeds, alpha_grid, title, sensitives):
    accuracy_our = {}
    accuracy_base = {}
    acc_mean_our = {}
    acc_var_our = {}
    acc_mean_base = {}
    acc_var_base = {}
    for s in sensitives:
        acc_mean_our[s] = []
        acc_var_our[s] = []
        acc_mean_base[s] = []
        acc_var_base[s] = []
    for alpha in alpha_grid:
        for s in sensitives:
            accuracy_our[alpha] = {}
            accuracy_our[alpha]['per_seed{}'.format(s)] = []
            accuracy_base[alpha] = {}
            accuracy_base[alpha]['per_seed{}'.format(s)] = []
            for seed in seeds:
                    accuracy_our[alpha]['per_seed{}'.format(s)].append(results[alpha, seed]['our']['acc{}'.format(s)])
                    accuracy_base[alpha]['per_seed{}'.format(s)].append(results[alpha, seed]['base']['acc{}'.format(s)])
            accuracy_our[alpha]['mean{}'.format(s)] = np.mean(accuracy_our[alpha]['per_seed{}'.format(s)])
            accuracy_our[alpha]['var{}'.format(s)] = np.var(accuracy_our[alpha]['per_seed{}'.format(s)])

            acc_mean_our[s].append(accuracy_our[alpha]['mean{}'.format(s)])
            acc_var_our[s].append(accuracy_our[alpha]['var{}'.format(s)])

            accuracy_base[alpha]['mean{}'.format(s)] = np.mean(accuracy_base[alpha]['per_seed{}'.format(s)])
            accuracy_base[alpha]['var{}'.format(s)] = np.var(accuracy_base[alpha]['per_seed{}'.format(s)])

            acc_mean_base[s].append(accuracy_base[alpha]['mean{}'.format(s)])
            acc_var_base[s].append(accuracy_base[alpha]['var{}'.format(s)])


    plt.figure('Accuracy')
    plt.title(title)
    for s in sensitives:
        plt.plot(alpha_grid, acc_mean_our[s], label='Our: s={}'.format(s))
        plt.fill_between(alpha_grid,
                         acc_mean_our[s] - np.sqrt(acc_var_our[s]),
                         acc_mean_our[s] + np.sqrt(acc_var_our[s]),
                         alpha=0.3)
        plt.plot(alpha_grid, acc_mean_base[s], label='Base: s={}'.format(s))
        plt.fill_between(alpha_grid,
                         acc_mean_base[s] - np.sqrt(acc_var_base[s]),
                         acc_mean_base[s] + np.sqrt(acc_var_base[s]),
                         alpha=0.3)
    plt.legend()
    plt.xlabel('alpha')
    plt.xlim(0.499, 0.951)
    plt.ylabel('accuracy')


def positive_vs_alpha_per_group(results, seeds, alpha_grid, title, sensitives):
    accuracy_our = {}
    accuracy_base = {}
    acc_mean_our = {}
    acc_var_our = {}
    acc_mean_base = {}
    acc_var_base = {}
    for s in sensitives:
        acc_mean_our[s] = []
        acc_var_our[s] = []
        acc_mean_base[s] = []
        acc_var_base[s] = []
    for alpha in alpha_grid:
        for s in sensitives:
            accuracy_our[alpha] = {}
            accuracy_our[alpha]['per_seed{}'.format(s)] = []
            accuracy_base[alpha] = {}
            accuracy_base[alpha]['per_seed{}'.format(s)] = []
            for seed in seeds:
                    accuracy_our[alpha]['per_seed{}'.format(s)].append(results[alpha, seed]['our']['pos{}'.format(s)])
                    accuracy_base[alpha]['per_seed{}'.format(s)].append(results[alpha, seed]['base']['pos{}'.format(s)])
            accuracy_our[alpha]['mean{}'.format(s)] = np.mean(accuracy_our[alpha]['per_seed{}'.format(s)])
            accuracy_our[alpha]['var{}'.format(s)] = np.var(accuracy_our[alpha]['per_seed{}'.format(s)])

            acc_mean_our[s].append(accuracy_our[alpha]['mean{}'.format(s)])
            acc_var_our[s].append(accuracy_our[alpha]['var{}'.format(s)])

            accuracy_base[alpha]['mean{}'.format(s)] = np.mean(accuracy_base[alpha]['per_seed{}'.format(s)])
            accuracy_base[alpha]['var{}'.format(s)] = np.var(accuracy_base[alpha]['per_seed{}'.format(s)])

            acc_mean_base[s].append(accuracy_base[alpha]['mean{}'.format(s)])
            acc_var_base[s].append(accuracy_base[alpha]['var{}'.format(s)])


    plt.figure('Positive')
    plt.title(title)
    for s in sensitives:
        plt.plot(alpha_grid, acc_mean_our[s], label='Our: s={}'.format(s))
        plt.fill_between(alpha_grid,
                         acc_mean_our[s] - np.sqrt(acc_var_our[s]),
                         acc_mean_our[s] + np.sqrt(acc_var_our[s]),
                         alpha=0.3)
        plt.plot(alpha_grid, acc_mean_base[s], label='Base: s={}'.format(s))
        plt.fill_between(alpha_grid,
                         acc_mean_base[s] - np.sqrt(acc_var_base[s]),
                         acc_mean_base[s] + np.sqrt(acc_var_base[s]),
                         alpha=0.3)
    plt.legend()
    plt.xlabel('alpha')
    plt.xlim(0.499, 0.951)
    plt.ylabel('positive rate')

NAME = 'Adult_RF'

seeds = np.arange(20)
alphas_grid = np.linspace(.5, .95, 10)

results = load_obj(NAME)

accuracy_vs_alpha(results, seeds, alphas_grid,
                  title='[Acc] Dataset: Adult. Method: LR')

accuracy_vs_alpha_per_group(results, seeds, alphas_grid, title='[GrAcc] Dataset: Adult. Method: LR', sensitives=[0.0, 1.0])


positive_vs_alpha_per_group(results, seeds, alphas_grid, title='[DP] Dataset: Adult. Method: LR', sensitives=[0.0, 1.0])


reject_vs_alpha_per_group(results, seeds, alphas_grid, title='[Reject] Dataset: Adult. Method: LR', sensitives=[0.0, 1.0])