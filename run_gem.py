# -*- coding: utf-8 -*-
import torch
import argparse
import logging
import csqn
import baselines
import model
from train_functions import *
from dataset import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
    Main file to run the desired Continual Learning experiments with one of the possible CL methods.

    To determine hyper-parameters such as the weight of the regularization in regularization-based methods, we use 
    the methodology of (Lopez-Paz et al., 2017), in which the weight is determined based on cross-validation over 
    the first T_CV tasks. 

    References: 
    David Lopez-Paz and Marc'Aurelio Ranzato. 2017. Gradient episodic memory for continual learning. 
        In Proceedings of the 31st International Conference on Neural Information Processing Systems (NIPS'17). 
        Curran Associates Inc., Red Hook, NY, USA, 6470–6479.
"""

""" Parsing command line arguments """
parser = argparse.ArgumentParser(description='Continual Learning experiments..')
parser.add_argument('--experiments', type=int, default=0,
                    help="Choose 0 for Rotated MNIST (default), 1 for Split CIFAR-10/100, 2 for Five Datasets")
parser.add_argument('--method', type=str, default='Fine-Tuning',
                    help="Select the CL method. Choose from Fine-Tuning (default), EWC, MAS, LWF, OGD and "
                         "CSQN-X (M) with X = S or B and M a positive integer. "
                         "Optionally, you can add to CSQN-X (M) R with R in [2N, EV, CT, SPL] as reduction strategy.")
parser.add_argument('--start', type=int, default=1, help='ID of first iteration')
parser.add_argument('--iter', type=int, default=3, help='Number of iterations')
parser.add_argument('--tasks', type=int, default=20,
                    help='Number of tasks to run. Only used when Rotated MNIST experiments are selected')
parser.add_argument('--angle', type=int, default=5,
                    help='Angle of the rotation. Only used when Rotated MNIST experiments are selected')
parser.add_argument('--log_file', type=str, default='log',
                    help='Name of the log file - default is just log. If T, results are written to Terminal')
parser.add_argument('--TCV', type=int, default=5, help='Number of tasks in CV to determine optimal hyper-parameters')
args = parser.parse_args()


""" Logging """
FORMAT = '%(asctime)-15s %(message)s'
if args.log_file == 'T':
    logging.basicConfig(format=FORMAT,  level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S %p')
else:
    logging.basicConfig(filename=args.log_file, format=FORMAT,  level=logging.DEBUG, datefmt='%m/%d/%Y %I:%M:%S %p')
logger = logging.getLogger('main')

logging.info("Set the device: %s", str(device))


""" Hyper-parameter search settings """
T_CV = args.TCV
logger.debug("Methodology of (Lopez-Paz et al., 2017) with T_CV = %d" % T_CV)
lambdas = {'EWC': [1, 100, 1000, 10000, 100000],
           'MAS': [1, 10, 100, 1000, 10000],
           'LWF': [0.01, 0.1, 1, 10, 100],
           'CSQN': [1, 100, 1000, 10000, 100000]}

""" Number of runs """
start_id, iterations = args.start, args.iter

""" Load the data and the network. Initialize train_functions """
if args.experiments == 0:
    n_tasks, offset, batch_size, n_epoch = args.tasks, args.angle, 64, 10
    train_data, dev_data, test_data = get_rotated_mnist_data(n_tasks=n_tasks, angle=offset, batch_size=batch_size)
    train_data1, _, _ = get_rotated_mnist_data(n_tasks=n_tasks, angle=offset, batch_size=1)
    get_net = lambda: model.get_net('mlp', 10)
    shared_layers = None
    first_task = 'task1_rotmnist_mlp'
    res_name = 'rotated_mnist_results_%d_%d_lange' % (offset, n_tasks)
    init(train_data, dev_data, test_data, res_name, n_tasks, n_epoch)
    n_classes = None
    sample_data = train_data
    first_epoch = n_epoch
elif args.experiments == 1:
    n_tasks, batch_size, n_epoch = 11, 64, 10
    train_data, dev_data, test_data, n_classes = get_split_cifar10_100_data(n_tasks=n_tasks-1, batch_size=batch_size)
    train_data1, _, _, _ = get_split_cifar10_100_data(n_tasks=n_tasks-1, batch_size=1)
    _, shared_layers = model.get_net('resnet', [n_classes] * (n_tasks + 1), return_shared_layers=True)
    get_net = lambda: model.get_net('resnet', [n_classes] * (n_tasks + 1))
    res_name = 'cifar10100_resnet_results_1'
    first_task = 'task1_resnet_cifar10'
    init(train_data, dev_data, test_data, res_name, n_tasks, n_epoch, n_classes, shared_layers)
    sample_data = train_data
    first_epoch = 15
elif args.experiments == 2:
    batch_size, n_epoch, n_tasks = 128, 50, 5
    n_classes = 10
    train_data, dev_data, test_data, sample_data = get_five_datasets(batch_size=batch_size)
    train_data1, _, _, _ = get_five_datasets(batch_size=1)
    get_net = lambda: model.get_net('lenet', [n_classes] * n_tasks)
    _, shared_layers = model.get_net('lenet', [n_classes] * n_tasks, return_shared_layers=True)
    res_name = 'five_dataset_results_1'
    first_task = 'first_task_resnet_five'
    init(train_data, dev_data, test_data, res_name, n_tasks, n_epoch, n_classes, shared_layers)
    first_epoch = n_epoch
else:
    raise Exception('Please choose 0, 1 or 2 for --experiments')

""" T_CV cannot be larger than the number of tasks """
T_CV = min(T_CV, n_tasks)

""" Printing information related to neural network """
net = get_net()
logging.debug("Neural network contains " + str(torch.cat([p.view(-1) for p in list(net.parameters())]).numel())
              + " parameters.")
for n, p in net.state_dict().items():
    logging.debug(n + " ---> %d parameters" % (p.numel()))

""" Loading the model trained on task 1 """
net_one = get_net()
try:
    net_one.load_state_dict(torch.load(first_task))
except Exception as e:  # at this point we assume it did not exist, so we train a model from scratch on task 1
    logger.warning('Exception: file %s did not exist - training on task one.. (exception was = %s)' % (first_task, e))
    train_net(net_one, 0, epochs=first_epoch, eval=True)
    torch.save(net_one.state_dict(), first_task)


""" Determining the method.. """
if 'CSQN' in args.method:
    method = 'CSQN'
    qnm = 'BFGS' if args.method.split(' ')[0].split('-')[-1] == 'B' else 'SR1'
    M = int(args.method.split(' ')[1].strip('(').strip(')'))
    if len(args.method.split(' ')) > 2:
        red = {'EV': 'auto', '2N': 'tree', 'CT': 'm', 'SPL': 'split'}
        reduction = red[args.method.split(' ')[-1]]
    else:
        reduction = 'none'
else:
    method = args.method
    qnm, reduction, M = None, None, None


logger.debug('Running with CL method: %s' % method)


""" To initialize the regulator """
choose_regulator = {'EWC': baselines.EWC, 'MAS': baselines.MAS, 'OGD': baselines.OGD,
                    'LWF': baselines.LWF, 'CSQN': csqn.CSQN}
arguments = {'EWC': (shared_layers, n_classes), 'MAS': (shared_layers,), 'LWF': (),
             'OGD': (shared_layers, 200, n_classes),
             'CSQN': (qnm, shared_layers, M, reduction, n_classes)}

logger.debug('Step 1: hyper-parameter search')
best_alpha = 0
if method in ['EWC', 'MAS', 'LWF', 'CSQN']:  # only required for methods which do have a regularization weight

    best_avg = 0
    regulator = choose_regulator[method](*arguments[method])

    for alpha in lambdas[method]:

        net_ = get_net()
        net_.load_state_dict(net_one.state_dict())

        for task in range(T_CV):

            logger.info("Adapting Task %d to Task %d" % (task, task + 1))

            if method == 'EWC':
                regulator.compute_FIM(net_, sample_data(task),
                                      task if shared_layers is not None else None)
            elif method == 'MAS':
                regulator.compute_IW(net_, sample_data(task),
                                     task if shared_layers is not None else None)
            elif method == 'LWF':
                regulator.set_old_net(net_)
                if shared_layers is not None:
                    logger.debug('Method = LWF, running training non-shared layers..')
                    train_net(net_, task + 1, epochs=n_epoch // 2, freeze_layers=shared_layers)
            elif method == 'CSQN':
                regulator.update(net_, sample_data(task), task if shared_layers is not None else None)

            logger.debug('alpha = %s' % alpha)
            if method in ['EWC', 'MAS', 'CSQN']:
                train_net(net_, task + 1, reg_loss=lambda x: regulator.regularize(x, alpha), epochs=n_epoch)
            else:
                if shared_layers is None:
                    train_net(net_, task + 1, reg_loss=lambda x, y: regulator.regularize(x, y, alpha), epochs=n_epoch)
                else:
                    train_net(net_, task + 1, reg_loss=lambda x, y, t: regulator.regularize(x, y, alpha, t),
                              epochs=n_epoch)
        names, accs = [('Task %d' % i) for i in range(T_CV)], [round(test(net_, i, 'dev', print_result=False), 2)
                                                               for i in range(T_CV)]
        logger.info('Final model: ' + str(names) + " = " + str(accs))
        avg_acc = sum(accs) / len(accs)
        if avg_acc > best_avg:
            best_avg = avg_acc
            best_alpha = alpha

    opt_alpha = best_alpha


logger.info("")
logger.debug('Step 2: The experiments with the optimal regularization weight...')


for num in range(start_id, start_id + iterations):

    alpha = best_alpha

    mod_list = [copy.deepcopy(net_one).state_dict()]

    logger.info("### RUN %d of %d " % (num, iterations))
    if method != 'Fine-Tuning':
        regulator = choose_regulator[method](*arguments[method])

    for task in range(n_tasks - 1):

        logger.info("Adapting Task %d to Task %d" % (task, task + 1))
        net_ = get_net()
        net_.load_state_dict(mod_list[-1])
        names, accs = [('Task %d' % i) for i in range(task + 1)], [round(test(net_, i, 'dev', print_result=False), 2)
                                                                   for i in range(task + 1)]
        logger.info('Initial model: ' + str(names) + " = " + str(accs))

        if method == 'EWC':
            regulator.compute_FIM(copy.deepcopy(net_), sample_data(task), task if shared_layers is not None else None)
        elif method == 'MAS':
            regulator.compute_IW(copy.deepcopy(net_), sample_data(task), task if shared_layers is not None else None)
        elif method == 'OGD':
            regulator.compute_gradients(copy.deepcopy(net_), train_data1(task),
                                        task if shared_layers is not None else None)
        elif method == 'LWF':
            regulator.set_old_net(net_)
            if shared_layers is not None:
                logger.debug('Method = LWF, running training non-shared layers..')
                train_net(net_, task + 1, epochs=n_epoch // 2, freeze_layers=shared_layers)
        elif method == 'CSQN':
            regulator.update(copy.deepcopy(net_), sample_data(task), task if shared_layers is not None else None)

        if method not in ['OGD']:

            logger.info('[Learning task %d with alpha = %s]' % (task, str(alpha)))
            if method in ['EWC', 'MAS', 'CSQN']:
                train_net(net_, task + 1, reg_loss=lambda x: regulator.regularize(x, alpha), epochs=n_epoch)
            else:
                if shared_layers is None:
                    train_net(net_, task + 1,  reg_loss=lambda x, y: regulator.regularize(x, y, alpha), epochs=n_epoch)
                else:
                    train_net(net_, task + 1, reg_loss=lambda x, y, t: regulator.regularize(x, y, alpha, t),
                              epochs=n_epoch)
            acc = test(net_, task + 1, 'dev')
            logger.info("[Task %d] = [%.2f]" % (task + 1, acc))

            best_alpha = alpha / a_hyp
            logger.info("Finished: best_alpha = %s - with accuracy of %.2f" % (str(best_alpha), acc))
            mod_list.append(net_.state_dict())
        else:
            train_net(net_, task + 1, grad_fn=lambda x: regulator.regularize(x), epochs=n_epoch, opt='sgd')
            mod_list.append(net_.state_dict())

    logger.info("Step 3: Evaluation")
    net_list = []
    for mod in mod_list:
        net_ = get_net().cpu()
        net_.load_state_dict(mod)
        net_list.append(net_)
    if method == 'Fine-Tuning':
        test_and_update(net_list, method)
        break
    test_and_update(net_list, method + ' %d' % num)
