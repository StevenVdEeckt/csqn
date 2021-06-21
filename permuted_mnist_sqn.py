# -*- coding: utf-8 -*-
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


parser = argparse.ArgumentParser(description='Permuted MNIST experiments..')
parser.add_argument('--method', type=str, default="SR1", help='BFGS or SR1 - default SR1')
parser.add_argument('--m', type=int, default=10, help='Number of components - default 10')
parser.add_argument('--eps_ewc', type=float, default=1e-4, help='epsilon to compute inverse FIM - default is 1e-4')
parser.add_argument('--start', type=int, default=1, help='ID of first iteration - default 1')
parser.add_argument('--iter', type=int, default=3, help='Number of iterations - default 3')
parser.add_argument('--reduction', type=str, default='none', help='How to reduce Z - default none')
parser.add_argument('--tasks', type=int, default=10, help='Number of tasks to run - default 10')
parser.add_argument('--log_id', type=int, default=0, help='ID of log file, to assure it is unique - default 0')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs - default is 5')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size of the training data - default is 64')
parser.add_argument('--p_hyp', type=float, default=0.8,
                    help='p-value for hyper-parameter search, see De Lange et al., 2019 - default is 0.8')
parser.add_argument('--a_hyp', type=float, default=0.5,
                    help='alpha for hyper-parameter search, see De Lange et al., 2019 - default is 0.5')


args = parser.parse_args()

""" Logging """
FORMAT = '%(asctime)-15s %(message)s'
if args.log_id == -1:
    logging.basicConfig(format=FORMAT, level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S %p')
else:
    logging.basicConfig(filename='perm_mnist_sqn_%d.log' % args.log_id, format=FORMAT,  level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger('main')

logging.info("Set the device: %s", str(device))

""" SQN Settings """
M, method = args.m, args.method  # 'SR1' or 'BFGS'
start_id, iterations = args.start, args.iter
reduction = args.reduction  # to reduce the number of components to store - options: 'none', 'auto', 'split', 'tree'
eps_ewc = args.eps_ewc  # to compute the inverse FIM

""" Hyper-parameter search settings """
p_hyp = args.p_hyp  # method must at least reach p*ACC_FT on new task
a_hyp = args.a_hyp  # decaying factor

""" Experiment settings """
n_classes = 10
n_tasks = args.tasks
n_epoch = args.epochs
batch_size = args.batch_size
first_task = 'task1_permmnist_mlp'
res_name = 'permuted_mnist_results_%d' % (n_tasks)
init_alpha = 100000 * 20 // M


""" Load the data """
train_data, dev_data, test_data = get_permuted_mnist_data(n_tasks=n_tasks, batch_size=batch_size)
""" Load the network """
Net = lambda: get_net('mlp', n_classes)

net = Net()
logging.debug("Neural network contains " + str(torch.cat([p.view(-1) for p in list(net.parameters())]).numel())
      + " parameters.")
for n, p in net.state_dict().items():
    logging.debug(n + " ---> %d parameters" % (p.numel()))


init(train_data, dev_data, test_data, res_name, n_tasks, n_epoch)  # initialize train_functions module


net_one = Net()
net_one.load_state_dict(torch.load(first_task))


logger.info("")
logger.info("# 3 Experiments with CL with SQN Regularization")


for num in range(start_id, start_id + iterations):

    mod_list = [copy.deepcopy(net_one).state_dict()]  # mod_list contains the state_dicts after training each task

    logger.info("### RUN %d of %d " % (num, start_id + iterations))
    c_sqn = CSQN(method=method, M=M, eps_ewc=eps_ewc, reduction=reduction)  # initialize CSQN object

    for task in range(n_tasks - 1):

        logger.info("Adapting Task %d to Task %d" % (task, task + 1))
        net_ = Net()
        net_.load_state_dict(mod_list[-1])
        names, accs = [('Task %d' % i) for i in range(task + 1)], [round(test(net_, i, 'dev', print_result=False), 2)
                                                                   for i in range(task + 1)]
        logger.debug('Initial model: ' + str(names) + " = " + str(accs))

        c_sqn.update(copy.deepcopy(net_), train_data(task))

        logging.info("Adapting without regularization..")
        train_net(net_, task + 1, epochs=n_epoch)  # we first adapt without regularization, obtaining accuracy A
        acc_ft = test(net_, task + 1, 'dev')

        acc, alpha = 0, init_alpha  # we start with a very high lambda (here called alpha)

        while acc < p_hyp * acc_ft:  # as long as the accuracy is < 0.8*A, we decrease alpha
            logger.info(c_sqn.get_name() + '-T%d [%d]' % (task, alpha))
            net_ = Net()
            net_.load_state_dict(mod_list[-1])
            names, accs = [('Task %d' % i) for i in range(task + 1)], [round(test(net_, i, 'dev', print_result=False), 2)
                                                                       for i in range(task + 1)]
            logger.debug('Initial model: ' + str(names) + " = " + str(accs))
            train_net(net_, task + 1, reg_loss=lambda x: c_sqn.regularize(x, alpha), epochs=n_epoch)
            acc = test(net_, task + 1, 'dev')
            logger.info("[Task %d] = [%.2f]" % (task + 1, acc))
            alpha = a_hyp * alpha

        best_alpha = alpha / a_hyp
        logger.debug("Best alpha was %d - with an average accuracy of %.2f" % (best_alpha, acc))

        mod_list.append(net_.state_dict())  # add it to mod_list and repeat for the next task

    logger.info("## 3.3 Evaluation")
    logger.info(c_sqn.get_name())
    net_list = []
    for mod in mod_list:
        net_ = Net().cpu()
        net_.load_state_dict(mod)
        net_list.append(net_)
    test_and_update(net_list, c_sqn.get_name() + ' %d' % num)  # evaluate the models and add to the results dict
