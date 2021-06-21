# -*- coding: utf-8 -*-
"""Copy of Reduction_strategies_for_CIFAR10_ClassIncremental_SR1_experiments.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1B0Zyh6_gUoVytfH1IpDGHbE0JrKNQCDr

# CIFAR-100 Class-Incremental Experiments with S-LSR1

Tasks: Ten CIFAR-100 tasks, each with 10 of the 100 classes. Each task has a separate classification layer.

Model: 2 conv layers + 2 linear layers, 62k parameters in total.
"""

import torch
from dataset import *
import model
import argparse
from train_functions import *
import logging
from csqn_inv import CSQN_Inv
from dataset import *

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger('main')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.debug('Device is: %s', device)


parser = argparse.ArgumentParser(description='Split CIFAR-100 Baselines..')
parser.add_argument('--method', type=str, default='SR1', help='Method to use for CiSQN')
parser.add_argument('--m', type=int, default=10, help='Number of components')
parser.add_argument('--start', type=int, default=1, help='ID of first iteration')
parser.add_argument('--iter', type=int, default=5, help='Number of iterations')
parser.add_argument('--reduction', type=str, default='none', help='How to reduce Z')
parser.add_argument('--tasks', type=int, default=10, help='Number of tasks to run')

args = parser.parse_args()

""" SQN Settings """
M, method = args.m, args.method  # 'SR1' or 'BFGS'
start_id, iterations = args.start, args.iter
eps_ewc = 1e-4

""" Experimental settings """
n_epoch = 5
n_tasks = args.tasks
batch_size = 64
res_name = 'cifar100_results_%d' % n_tasks
first_task = 'task1_cifar100_%d' % (100 // n_tasks)

""" Loading the data"""
train_data, dev_data, test_data, n_classes = get_split_cifar100_data(n_tasks, batch_size)

Net = lambda: model.get_net('lenet-small', [n_classes] * n_tasks)
net, shared_layers = model.get_net('lenet-small', [n_classes] * n_tasks, return_shared_layers=True)

logger.debug("Neural network contains " + str(torch.cat([p.view(-1) for p in list(net.parameters())]).numel())
             + " parameters.")
for n, p in net.named_parameters():
    logger.debug(n + ' contains %d parameters' % (p.numel()))


init(train_data, dev_data, test_data, res_name, n_tasks, n_epoch, n_classes, shared_layers)

net_one = Net()
net_one.load_state_dict(torch.load(first_task))


logger.info("")
logger.info("# 3 Experiments with CL with SQN Regularization")



mod_list = [copy.deepcopy(net_one).state_dict()]

c_sqn = CSQN_Inv(method=method, shared_layers=shared_layers, M=M, eps_ewc=eps_ewc, n_classes=n_classes)

num = 0

for task in range(n_tasks - 1):
    logger.info("Adapting Task %d to Task %d" % (task, task + 1))
    net_ = Net()
    net_.load_state_dict(mod_list[-1])
    names, accs = [('Task %d' % i) for i in range(task + 1)], [round(test(net_, i, 'dev', print_result=False), 2)
                                                               for i in range(task + 1)]
    logger.info('Initial model: ' + str(names) + " = " + str(accs))

    c_sqn.update(copy.deepcopy(net_), train_data(task), task)

    net_ = Net()
    net_.load_state_dict(mod_list[-1])
    train_net(net_, task + 1, grad_fn=lambda x: c_sqn.regularize(x), epochs=n_epoch, opt='sgd')
    acc = test(net_, task + 1, 'dev')
    names, accs = [('Task %d' % i) for i in range(task + 2)], [round(test(net_, i, 'dev', print_result=False), 2)
                                                               for i in range(task + 2)]
    logger.info(str(names) + " = " + str(accs))

    mod_list.append(net_.state_dict())

logger.info("## 3.3 Evaluation")
logger.info(c_sqn.get_name())
net_list = []
for mod in mod_list:
    net_ = Net().cpu()
    net_.load_state_dict(mod)
    net_list.append(net_)
test_and_update(net_list, c_sqn.get_name() + ' %d' % num, add=False)

