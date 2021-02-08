import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib import rc
import seaborn as sns
import os

# Set it to false if you do not want to render plots with your own tex.
MY_LATEX = True
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('text.latex',
       preamble=r'\usepackage{mathtools}\usepackage{amsmath,amsfonts}')
params = {'text.usetex' : True,
          'font.size' : 13,
          'font.family' : 'lmodern',
          'text.latex.unicode': True,
          }
plt.rcParams["mathtext.fontset"] = 'dejavusans'

if MY_LATEX:
    plt.rcParams['text.usetex'] = True
a = sns.color_palette("colorblind")
a = [a[0], a[2], a[3], a[4]]

def load_obj(name):
    with open('results/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)



def accuracy_vs_alpha(results, seeds, alpha_grid, title, lb, ub, name):
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


    plt.figure('Accuracy Total', figsize=(7, 7))
    plt.title(title)
    plt.plot(alpha_grid, acc_mean_our, label=r'Our', color=a[0])
    plt.fill_between(alpha_grid,
                     acc_mean_our - np.sqrt(acc_var_our),
                     acc_mean_our + np.sqrt(acc_var_our),
                     alpha=0.3, color=a[0], linestyle='-')
    plt.plot(alpha_grid, acc_mean_base, label=r'Base', color=a[2])
    plt.fill_between(alpha_grid,
                     acc_mean_base - np.sqrt(acc_var_base),
                     acc_mean_base + np.sqrt(acc_var_base),
                     alpha=0.3, color=a[2])
    plt.legend()
    plt.xlabel(r'\Large$\alpha$')
    plt.xlim(0.501, 0.949)
    plt.ylabel(r'\large Accuracy')
    plt.ylim((0.99)* lb, (1.01) * ub)
    plt.savefig('results/plots/acc_{}.pdf'.format(name), bbox_inches='tight')



def reject_vs_alpha_per_group(results, seeds, alpha_grid,
                              title, sensitives, lines, name):
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


    plt.figure('Reject', figsize=(7, 7))
    plt.title(title)
    for s in sensitives:
        plt.plot(alpha_grid, acc_mean_our[s], label=r'Our: \Large$s={}$'.format(int(s)),
                color=a[0], linestyle=lines[s])
        plt.fill_between(alpha_grid,
                         acc_mean_our[s] - np.sqrt(acc_var_our[s]),
                         acc_mean_our[s] + np.sqrt(acc_var_our[s]),
                         alpha=0.3, color=a[0], linestyle=lines[s])
    plt.legend()
    plt.xlabel(r'\Large $\alpha$')
    plt.xlim(0.499, 0.951)
    plt.ylim(0.499, 0.951)
    plt.ylabel(r'\large Classification rate')
    plt.grid()
    plt.savefig('results/plots/reject_{}.pdf'.format(name),
                bbox_inches='tight')




def accuracy_vs_alpha_per_group(results, seeds, alpha_grid,
                                title, sensitives, lines, name):
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


    plt.figure('AccuracyG', figsize=(7, 7))
    plt.title(title)
    ub = 0
    lb = 1
    for s in sensitives:
        plt.plot(alpha_grid, acc_mean_our[s], label=r'Our: \Large$s={}$'.format(int(s)), color=a[0], linestyle=lines[s])
        tmp = np.max(acc_mean_our[s] + np.sqrt(acc_var_our[s]))
        ub = tmp if tmp > ub else ub
        tmp = np.min(acc_mean_our[s] - np.sqrt(acc_var_our[s]))
        lb = tmp if tmp < lb else lb
        plt.fill_between(alpha_grid,
                         acc_mean_our[s] - np.sqrt(acc_var_our[s]),
                         acc_mean_our[s] + np.sqrt(acc_var_our[s]),
                         alpha=0.3, color=a[0], linestyle=lines[s])
        plt.plot(alpha_grid, acc_mean_base[s], label=r'Base: \Large$s={}$'.format(int(s)), color=a[2], linestyle=lines[s])
        plt.fill_between(alpha_grid,
                         acc_mean_base[s] - np.sqrt(acc_var_base[s]),
                         acc_mean_base[s] + np.sqrt(acc_var_base[s]),
                         alpha=0.3, color=a[2], linestyle=lines[s])
        tmp = np.max(acc_mean_base[s] + np.sqrt(acc_var_base[s]))
        ub = tmp if tmp > ub else ub
        tmp = np.min(acc_mean_base[s] - np.sqrt(acc_var_base[s]))
        lb = tmp if tmp < lb else lb
    plt.legend()
    plt.xlabel(r'\Large$\alpha$')
    plt.xlim(0.499, 0.951)
    plt.ylim((0.99)* lb, (1.01) * ub)
    plt.ylabel(r'\large Accuracy per Group')
    plt.savefig('results/plots/accg_{}.pdf'.format(name), bbox_inches='tight')
    return lb, ub


def positive_vs_alpha_per_group(results, seeds, alpha_grid,
                                title, sensitives, lines, name):
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


    plt.figure('Positive', figsize=(7, 7))
    plt.title(title)
    for s in sensitives:
        plt.plot(alpha_grid, acc_mean_our[s], label=r'Our: \Large$s={}$'.format(int(s)), color=a[0], linestyle=lines[s])
        plt.fill_between(alpha_grid,
                         acc_mean_our[s] - np.sqrt(acc_var_our[s]),
                         acc_mean_our[s] + np.sqrt(acc_var_our[s]),
                         alpha=0.3, color=a[0], linestyle=lines[s])
        plt.plot(alpha_grid, acc_mean_base[s], label=r'Base: \Large$s={}$'.format(int(s)), color=a[2], linestyle=lines[s])
        plt.fill_between(alpha_grid,
                         acc_mean_base[s] - np.sqrt(acc_var_base[s]),
                         acc_mean_base[s] + np.sqrt(acc_var_base[s]),
                         alpha=0.3, color=a[2], linestyle=lines[s])
    plt.legend()
    plt.xlabel(r'\large$\alpha$')
    plt.xlim(0.499, 0.951)
    plt.ylabel(r'\large Positive rate')
    plt.savefig('results/plots/pos_{}.pdf'.format(name), bbox_inches='tight')




DATA = "German"
METHOD = "LR"
NAME = '{}_{}'.format(DATA, METHOD)
SENSITIVES = [0.0, 1.0]
COLORS = {
         0.0 : a[0],
         1.0 : a[1]
         }
LINES = {
        0.0 : '-',
        1.0 : '--'
        }

title = r'Dataset: \textsc{{{}}}. Method: {}'.format(DATA, METHOD)


seeds = np.arange(20)
alphas_grid = np.linspace(.5, .95, 10)

results = load_obj(NAME)
if not os.path.exists('results/plots'):
    os.makedirs('results/plots')

lb, ub = accuracy_vs_alpha_per_group(results, seeds, alphas_grid,
                                     title=title, sensitives=SENSITIVES,
                                     lines=LINES, name=NAME)

accuracy_vs_alpha(results, seeds, alphas_grid,
                  title=title, lb=lb, ub=ub, name=NAME)

reject_vs_alpha_per_group(results, seeds, alphas_grid, title=title,
                          sensitives=SENSITIVES, lines=LINES,
                          name=NAME)

positive_vs_alpha_per_group(results, seeds, alphas_grid, title=title,
                            sensitives=SENSITIVES, lines=LINES, name=NAME)

plt.show()
