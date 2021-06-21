# -*- coding: utf-8 -*-
"""Rotated_MNIST_experiments.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BsYrz6Dx8s8ou1xDaByVYgalvS7fk9rk
"""

import torch
import logging
from baselines import *
from dataset import *
from train_functions import *
import model
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Rotated MNIST Baselines..')
parser.add_argument('--start', type=int, default=1, help='ID of first iteration')
parser.add_argument('--iter', type=int, default=3, help='Number of iterations to run each method')
parser.add_argument('--tasks', type=int, default=20, help='Number of tasks to run')
parser.add_argument('--angle', type=int, default=5, help="Angle of the rotation")
parser.add_argument('--log_id', type=int, default=0, help="ID of log file, to assure it is unique")

args = parser.parse_args()


""" The baselines and their settings """
run = ['KF']
start_id = {'KF': args.start}
iterations = {'KF': args.iter}
init_alpha = {'KF': 1e5}

""" Hyper-parameter search settings """
p_hyp = 0.8  # method must at least reach p*ACC_FT on new task
a_hyp = 0.5  # decaying factor

""" Experiment settings """
n_tasks = args.tasks
offset = args.angle
n_epoch = 10
batch_size = 64
first_task = 'task1_rotmnist_mlp'
res_name = 'rotated_mnist_results_%d_%d' % (offset, n_tasks)
n_classes = 10

FORMAT = '%(asctime)-15s %(message)s'

logging.basicConfig(filename='rot_mnist_kf_%d_%d_%d.log' % (offset, n_tasks, args.log_id), level=logging.DEBUG, format=FORMAT, datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger('main')

""" In case of GPM, we use a biasless model """
bias = not 'GPM' in run
if not bias:
    first_task += "_no_bias"


""" Load the data """
train_data, dev_data, test_data = get_rotated_mnist_data(n_tasks=n_tasks, angle=offset, batch_size=batch_size)
train_data1, _, _ = get_rotated_mnist_data(n_tasks=n_tasks, angle=offset, batch_size=1)
""" Load the network """
Net = lambda: model.get_net('mlp', n_classes, bias=bias)
net = Net()


logging.debug("Neural network contains " + str(torch.cat([p.view(-1) for p in list(net.parameters())]).numel())
      + " parameters.")
for n, p in net.state_dict().items():
    logging.debug(n + " ---> %d parameters" % (p.numel()))

init(train_data, dev_data, test_data, res_name, n_tasks, n_epoch)


net_one = Net()
try:
    net_one.load_state_dict(torch.load(first_task))
except Exception as e:
    logger.warning('Exception: file %s did not exist - training on task one.. (exception was = %s)' % (first_task, e))
    train_net(net_one, 0, epochs=n_epoch)
    torch.save(net_one.state_dict(), first_task)


logger.info("")
logger.info("# 3 Experiments with the Baselines")


for method in run:

    logger.info("Method: %s" % method)

    for num in range(start_id[method], start_id[method] + iterations[method]):

        mod_list = [copy.deepcopy(net_one.state_dict())]

        logger.info("### RUN %d of %d " % (num, iterations[method]))

        if method == 'KF':
            regulator = KF_EWC()

        for task in range(n_tasks - 1):

            logger.info("Adapting Task %d to Task %d" % (task, task + 1))
            net_ = Net()
            net_.load_state_dict(mod_list[-1])
            names, accs = [('Task %d' % i) for i in range(task + 1)], [round(test(net_, i, 'dev', print_result=False), 2)
                                                                       for i in range(task + 1)]
            logger.info('Initial model: ' + str(names) + " = " + str(accs))

            if method == 'KF':
                regulator.update_KF(net_, train_data(task))

            logger.debug("Adapting without regularization..")
            train_net(net_, task + 1, epochs=n_epoch)
            acc_ft = test(net_, task + 1, 'dev')
            if method == 'Fine-Tuning':
                mod_list.append(net_.state_dict())
                continue

            acc, alpha = 0, init_alpha[method]

            while acc < p_hyp * acc_ft:
                logger.debug('alpha = %s' % alpha)
                net_ = Net()
                net_.load_state_dict(mod_list[-1])
                train_net(net_, task + 1, reg_loss=lambda x: regulator.regularize(x, alpha), epochs=n_epoch)
                acc = test(net_, task + 1, 'dev')
                logger.info("[Task %d] = [%.2f]" % (task + 1, acc))
                alpha = a_hyp * alpha

            best_alpha = alpha / a_hyp
            logger.info("Best alpha was %d - with an average accuracy of %.2f" % (best_alpha, acc))

            mod_list.append(net_.state_dict())

        logger.info("## 3.3 Evaluation")
        net_list = []
        for mod in mod_list:
            net_ = Net().cpu()
            net_.load_state_dict(mod)
            net_list.append(net_)
        test_and_update(net_list, method + ' %d' % num)