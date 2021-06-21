# -*- coding: utf-8 -*-
"""Rotated_MNIST_experiments.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1BsYrz6Dx8s8ou1xDaByVYgalvS7fk9rk
"""

import torch
import argparse
import logging
from csqn import CSQN
from model import *
from train_functions import *
from dataset import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def cuda_overview():
    t = torch.cuda.get_device_properties(0).total_memory * 10.0 ** (-9)
    r = torch.cuda.memory_reserved(0) * 10.0 ** (-9)
    a = torch.cuda.memory_allocated(0) * 10.0 ** (-9)
    f = r - a  # free inside reserved
    logging.debug("CUDA memory - total: %.4f; reserved: %.4f; allocated: %.4f; available: %.4f" % (t, r, a, f))


parser = argparse.ArgumentParser(description='Rotated MNIST experiments..')
parser.add_argument('--method', type=str, default="SR1", help='BFGS or SR1')
parser.add_argument('--m', type=int, default=10, help='Number of components')
parser.add_argument('--start', type=int, default=1, help='ID of first iteration')
parser.add_argument('--iter', type=int, default=3, help='Number of iterations')
parser.add_argument('--reduction', type=str, default='none', help='How to reduce Z')
parser.add_argument('--tasks', type=int, default=20, help='Number of tasks to run')
parser.add_argument('--angle', type=int, default=5, help='Angle of the rotation')
parser.add_argument('--log_id', type=int, default=0, help='ID of log file, to assure it is unique')

args = parser.parse_args()

""" Logging """
FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(filename='rot_mnist_sqn_%d.log' % args.log_id, format=FORMAT,  level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger('main')

logging.info("Set the device: %s", str(device))

""" SQN Settings """
M, method = args.m, args.method  # 'SR1' or 'BFGS'
start_id, iterations = args.start, args.iter
consolidation = 'Z'  # 'Z' or 'XA'
reduction = args.reduction  # 'none', 'auto', 'split', 'tree' - only relevant if consolidation = 'Z'
eps_ewc = 1e-4

lambdas = [1, 100, 1000, 10000, 100000]
T_CV = 6

""" Experiment settings """
n_classes = 10
n_tasks = args.tasks
offset = args.angle
n_epoch = 10
batch_size = 64
first_task = 'task1_rotmnist_mlp'
res_name = 'rotated_mnist_results_%d_%d' % (offset, n_tasks)


""" Load the data """
train_data, dev_data, test_data = get_rotated_mnist_data(n_tasks=n_tasks, angle=offset, batch_size=batch_size)
""" Load the network """
Net = lambda: get_net('mlp', n_classes)

net = Net()
logging.debug("Neural network contains " + str(torch.cat([p.view(-1) for p in list(net.parameters())]).numel())
      + " parameters.")
for n, p in net.state_dict().items():
    logging.debug(n + " ---> %d parameters" % (p.numel()))


try:
    results = torch.load(res_name, map_location='cpu')
except Exeption as e:
    logger.debug('Exception when loading results: %s' % str(e))
    results = {}

init(train_data, dev_data, test_data, res_name, n_tasks, n_epoch)


net_one = Net()

net_one.load_state_dict(torch.load(first_task))

logger.debug("Hyper-parameter search")

best_avg, best_alpha = 0, 0

regulator = CSQN(method=method, M=M, eps_ewc=eps_ewc, reduction=reduction)

for alpha in lambdas:

    net_ = copy.deepcopy(net_one)

    for task in range(T_CV):

        logger.info("Adapting Task %d to Task %d" % (task, task + 1))

        regulator.update(net_, train_data(task))

        logger.debug('alpha = %s' % alpha)
        train_net(net_, task + 1, reg_loss=lambda x: regulator.regularize(x, alpha), epochs=n_epoch)
    names, accs = [('Task %d' % i) for i in range(T_CV)], [round(test(net_, i, 'dev', print_result=False), 2)
                                                               for i in range(T_CV)]
    logger.info('Final model: ' + str(names) + " = " + str(accs))
    avg_acc = sum(accs) / len(accs)
    if avg_acc > best_avg:
        best_avg = avg_acc
        best_alpha = alpha



logger.info("")
logger.info("# 3 Experiments with CL with SQN Regularization")


for num in range(start_id, start_id + iterations):

    mod_list = [copy.deepcopy(net_one).state_dict()]

    logger.info("### RUN %d of %d " % (num, iterations))
    c_sqn = CSQN(method=method, M=M, eps_ewc=eps_ewc, reduction=reduction)

    alpha = best_alpha

    for task in range(n_tasks - 1):

        logger.info("Adapting Task %d to Task %d" % (task, task + 1))
        net_ = Net()
        net_.load_state_dict(mod_list[-1])
        names, accs = [('Task %d' % i) for i in range(task + 1)], [round(test(net_, i, 'dev', print_result=False), 2)
                                                                   for i in range(task + 1)]
        logger.info('Initial model: ' + str(names) + " = " + str(accs))

        c_sqn.update(copy.deepcopy(net_), train_data(task))


        logger.info(c_sqn.get_name() + '-T%d [%d]' % (task, alpha))
        net_ = Net()
        net_.load_state_dict(mod_list[-1])
        train_net(net_, task + 1, reg_loss=lambda x: c_sqn.regularize(x, alpha), epochs=n_epoch)
        acc = test(net_, task + 1, 'dev')
        logger.info("[Task %d] = [%.2f]" % (task + 1, acc))


        mod_list.append(net_.state_dict())

    logger.info("## 3.3 Evaluation")
    logger.info(c_sqn.get_name())
    net_list = []
    for mod in mod_list:
        net_ = Net().cpu()
        net_.load_state_dict(mod)
        net_list.append(net_)
    test_and_update(net_list, c_sqn.get_name() + ' %d' % num)
