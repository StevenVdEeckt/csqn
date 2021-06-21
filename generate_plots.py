import torch
import matplotlib as mpl
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
font = {'weight': 'normal', 'size': 10}
mpl.rc('font', **font)
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT,  level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger('main')


"""
    Python file to generate plots from the dictionary containing the results of the experiments. 
"""


"""
  Reading command line arguments  
"""
parser = argparse.ArgumentParser(description='Generate the plots and tables to summarize the results..')
parser.add_argument('results', type=str, help='Name of the file of the results dictionary')
parser.add_argument('name', type=str, help='Name of the experiments, e.g. Split CIFAR-100 experiments')
parser.add_argument('--no_plot', type=str, default='Scratch T1,Scratch T1 + Freeze & Adapt',
                    help='Names of methods that must not be included in the plot or tables, separated by a comma')
parser.add_argument('--rename', type=int, default=1, help='Set to 1 if methods must be re-named first')
parser.add_argument('--map', type=str, default='plots/', help='Map in which to store the plots')

args = parser.parse_args()


""" Important parameters """
res_file = args.results
noplot = args.no_plot.split(',')
title = args.name
file_prefix = args.name.lower().replace('-', '').replace(' ', '').replace('/', '')
if not os.path.isdir(args.map):
    os.mkdir(args.map)


"""
  Returns true if given name is our method
  :param str name: the name
"""
def is_our_method(name):
    return 'S-LSR1' in name or 'S-LBFGS' in name or 'CSQN' in name


"""
  Renames the methods from e.g. S-LSR1-10 to CSQN-S (10)
  :param str org_name: original name, to be changed
"""
def re_name(org_name):
    name = 'CSQN'
    if 'S-LSR1' in org_name:
        name += '-S'
    elif 'S-LBFGS' in org_name:
        name += '-B'
    M = org_name.split('-')[-1]
    name += " (%s)" % M
    return name


"""
  Re-groups the given dictionary by putting the baselines first
  :param dict results: the results dictionary
"""
def re_group(results_):
    order = ['Fine-Tuning', 'EWC', 'MAS', 'OGD', 'LWF', 'CSQN-S (10)', 'CSQN-S (20)', 'CSQN-B (10)', 'CSQN-B (20)']
    new_results = {}
    for method in order:
        if method not in results_.keys():
            continue
        new_results[method] = results_[method]
    return new_results


"""
  Aggregates the results, i.e. multiple runs of the same method are averaged
  :param dict results: the results dictionary
  :param bool rename: (optional) True if methods must be renamed
"""
def get_aggregate_results(results_, rename=False):
    aggr_results_ = {}
    for n, p in results_.items():
        do_not_plot = False
        for no in noplot:
            if no in n:
                do_not_plot = True
                break
        if do_not_plot:
            continue
        try:
            int(n.split(" ")[-1])
            name = ' '.join(n.split(" ")[:-1])
            if is_our_method(name) and rename:
                name = re_name(name)
        except:
            name = n
        try:
            aggr_results_[name]['K'] += 1
            K = aggr_results_[name]['K']
            aggr_results_[name]['R'] = aggr_results_[name]['R'] * (K - 1) / K + p['R'] / K
        except:
            aggr_results_[name] = {'K': 1, 'R': p['R']}
    return re_group(aggr_results_)


"""
  Returns the number of methods and baselines in the dictionary
  :param dict results: the results dictionary
"""
def get_counts(results_):
    method, baseline = 0, 1
    for n, p in results_.items():
        if is_our_method(n):
            method += 1
        else:
            baseline += 1
    return method, baseline


"""
  Returns the accuracy and backward transfer for the given method
  :param torch.tensor R: T*T tensor with T the number of tasks and R[i,j] the accuracy on task j after learning task i
"""
def compute_acc_bwt(R):
    acc = R[-1, :].sum().item() / R.size(0)
    bwt = sum([(R[-1, i] - R[i, i]).item() for i in range(R.size(0))]) / (R.size(0) - 1)
    return acc, bwt


"""
  Returns a summary of the results dictionary, with for each method the accuracy and backward transfer
  :param dict results: the results dictionary
"""
def get_summary(results_):
    summary = {}
    for n, p in results_.items():
        acc, bwt = compute_acc_bwt(p['R'])
        summary[n] = [acc, bwt]
    return summary


"""
  Computes the average accuracy over time for the results dictionary
  :param dict results: the results dictionary
"""
def compute_accuracy_over_time(results):
    new = {}
    for n, p in results.items():
        new[n] = [p['R'][i, :i+1].sum().item() / (i + 1) for i in range(p['R'].size(0))]
    return new


"""
  Loading the files
"""
results = torch.load(res_file)
aggr_results = get_aggregate_results(results, args.rename == 1)
summary = get_summary(aggr_results)


""" 
  Preparing the color maps for the plots 
"""
cm_base = plt.get_cmap('hot')
cm = plt.get_cmap('cool')
COLORS, COLORS_BASE = get_counts(aggr_results)


""" 
  Makes a plot: the results contain 
  :param dict results: dictionary with key is the name and value are the results of the method
  :param str name: the name of the file to be saved
  :param str title: the title of the plot
  :param str xlabel: the label of the x-axis
  :param str ylabel: the label of the y-axis
"""
def make_plot(results, name, title, xlabel, ylabel):
    N = len(list(results.values())[0])
    kBASE, k = 0, 0
    for n, p in results.items():
        if is_our_method(n):
            plt.plot(np.arange(len(p)) + 1, p, label=n, color=cm(k / COLORS))
            k += 1
        else:
            plt.plot(np.arange(len(p)) + 1, p, label=n, color=cm_base(kBASE / COLORS_BASE))
            kBASE += 1
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(np.arange(N) + 1, np.arange(N) + 1)
    plt.title(title)
    plt.gcf().set_size_inches(10, 6)
    plt.legend(loc='best')
    plt.savefig(name)
    plt.clf()


"""
  Makes a plot with the average accuracy over time, i.e. as tasks are added
"""
logger.info("Generating accuracy plot..")
make_plot(compute_accuracy_over_time(aggr_results), name=args.map + file_prefix + "avg.png",
          title="title", xlabel="Task", ylabel="Average accuracy")
logger.info("Finished!")


"""
  Makes a bar plot
  :param dict results: dictionary with key is the name and value are the results of the method
  :param str name: the name of the file to be saved
  :param str title: the title of the plot
  :param str xlabel: the label of the x-axis
  :param str ylabel: the label of the y-axis

"""
def make_bar_plot(results, name, title, xlabel, ylabel):
    kBASE, k = 0, 0
    start, step = 0.7, 0.8 / (COLORS + COLORS_BASE)
    width = step * 0.8
    for n, p in results.items():
        if is_our_method(n):
            plt.bar(start + step * (k + kBASE), p, label=n, color=cm(k / COLORS), width=width)
            k += 1
        else:
            plt.bar(start + step * (k + kBASE), p, label=n, color=cm_base(kBASE / COLORS_BASE), width=width)
            kBASE += 1
    plt.ylabel(ylabel)
    plt.xticks(xlabel)
    plt.title(title)
    plt.gcf().set_size_inches(10, 6)
    plt.legend(loc='best')
    plt.savefig(name)
    plt.clf()


"""
  Makes a bar plot for the average accuracy
"""
logger.info("Generating accuracy bar plot..")
make_bar_plot({n: p[0] for n, p in summary.items()}, name=args.map + file_prefix + "avgbarplot.png",
              title=title, xlabel=[], ylabel="Average Accuracy")
logger.info("Finished!")


"""
  Makes a bar plot for the backward transfer
"""
logger.info("Generating backward transfer bar plot..")
make_bar_plot({n: p[1] for n, p in summary.items()}, name=args.map + file_prefix + "bwtbarplot.png",
              title=title, xlabel=[], ylabel="Backward Transfer")
logger.info("Finished!")


"""
  Print LaTeX code for Table
  :param dict results: the results dictionary
"""
def print_latex_code_table(results):
    print("\\midrule")
    line = False
    for n, p in results.items():
        if is_our_method(n) and not line:
            print("\midrule")
            line = True
        print("%s & %.2f & %.2f %s" % (n, p[0], p[1], r"\\"))
    print("\\bottomrule")


"""
  Print the LaTeX code for the Table
"""
logger.info("Generating LaTeX code..")
print_latex_code_table(summary)
logger.info("Finished!")
