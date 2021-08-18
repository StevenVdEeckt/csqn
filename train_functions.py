import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import logging
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = logging.getLogger('main')
logger.debug('Device in train_functions.py = %s' % str(device))

"""
    Module containing functions helpful for training and testing Continual Learning models
"""


"""
  initializes the train_functions module
  :param function train_: returns a dataloader for the training data given a task id 
  :param function dev_: returns a dataloader for the validation data given a task id 
  :param function test_: returns a dataloader for the test data given a task id 
  :param str results_name: string of file to store results
  :param int tasks: number of tasks to be learned
  :param int num_epochs: (optional) number of epochs to train - can also be passed directly to train function
  :param int num_classes: (optional) required for split experiments
  :param list shared_layers: (optional) required for split experiments with multi-head classifier
"""
def init(train_, dev_, test_, results_name, tasks, num_epochs=5, num_classes=None, shared_layers=None):
    global n_classes
    n_classes = num_classes
    global train_data, dev_data, test_data
    train_data, dev_data, test_data = train_, dev_, test_
    global res_name
    res_name = results_name
    global n_tasks
    n_tasks = tasks
    global n_epochs
    n_epochs = num_epochs
    global is_shared, multihead
    is_shared = lambda x: shared_layers is None or x in shared_layers
    multihead = shared_layers is not None


"""
    to test the network on the given task and dataset
    :param nn.Module net_: neural network
    :param int task: task id
    :param str dataset: (optional) dataset to test, default is test set
    :param bool print_result: (optional) print result or not - default is True
"""
def test(net_, task, dataset='test', print_result=True):
    correct, total = 0, 0
    net_ = net_.to(device)
    net_.eval()
    testloader = test_data(task) if dataset == 'test' else dev_data(task) if dataset == 'dev' else train_data(task)
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if n_classes is not None:
                labels = labels % n_classes
            outputs = net_(images.to(device)) if not multihead else net_(images.to(device), task)
            _, predicted = torch.max(outputs.data, 1)
            corr_pred = predicted == labels.to(device)
            total += labels.size(0)
            correct += corr_pred.sum().item()
    acc = 100.0 * correct / (1.0 * total)
    if print_result:
        logger.debug('Accuracy of the network on the ' + dataset + '-%d images: %.2f %%' % (task, acc))
    net_.cpu()
    return acc


"""
  test model and save the results
  :param nn.Module or list of nn.Module model_: network or list of networks to be evaluated
  :param str name: name to give the model when saving the results
  :param bool add: (optional) True if the model should be saved, default is True
  :param bool dev: (optional) True if evaluated on dev set instead of test set (in that case results are not added)
"""
def test_and_update(model_, name, add=True, dev=False):
    R = torch.zeros([n_tasks, n_tasks])
    if not isinstance(model_, list):
        for n in range(n_tasks):
            acc = test(model_, n, dataset='dev' if dev else 'test')
            R[-1, n] = acc
        accs = {'Task ' + str(n): R[-1, n].item() for n in range(n_tasks)}
        avg_acc = sum(accs.values()) / len(accs)
        logger.debug("Average accuracy on the test images: %.2f %%" % avg_acc)
    else:
        k = 0
        for net_ in model_:
            logger.debug("Accuracies after training on task %d" % k)
            for n in range(n_tasks):
                acc = test(net_, n, dataset='dev' if dev else 'test')
                R[k, n] = acc
            k += 1
        accs = {'Task ' + str(n): R[-1, n].item() for n in range(n_tasks)}
        avg_acc = sum(accs.values()) / len(accs)
        logger.debug("Average accuracy on the test images: %.2f %%" % avg_acc)
    if add and not dev:
        try:
            results = torch.load(res_name)
        except Exception as e:
            logger.warning('Exception: %s' % str(e))
            results = {}
        accs['Average'] = avg_acc
        accs['R'] = R
        results[name] = accs
        torch.save(results, res_name)


"""
  transform dictionary or list of paramaters into 1D torch tensor
  :param dict or list param: dictionary or list of tensors
"""
def param_to_x(param):
    if isinstance(param, dict):
        return torch.cat([p.view(-1) for n, p in param.items()])
    elif isinstance(param, list) or isinstance(param, tuple):
        return torch.cat([p.reshape(-1) for p in param])
    else:
        raise Exception("expected dictionary, list or tuple")


"""
   transforms 1D torch tensor into a dictionary of tensors
   :param torch.tensor x: 1D tensor
   :param dict param_sizes: dictionary with for each key in the new dictionary and value 'total' and 'size'
"""
def x_to_params(x, param_sizes):
    k = 0
    param = {}
    for n, p in param_sizes.items():
        param[n] = x[k:k + p['total']].view(p['size'])
        k += p['total']
    return param


"""
   Updates the gradients of the model - used for methods such as OGD
   :param nn.Module model: the neural network
   :param function grad_fn: function to be applied to 1D tensor of gradients of model
"""
def update_gradients(model, grad_fn):
    grads = {}
    sizes = {}
    # we collect the gradients in a dict
    for n, p in model.named_parameters():
        if is_shared(n):
            try:
                grads[n] = 1.0 * p.grad
            except:
                grads[n] = torch.zeros_like(p)
            sizes[n] = {'total': p.numel(), 'size': p.size()}
    # we convert the dict to one vector
    x = param_to_x(grads)
    # we update the vector using grad_fn
    x_ = grad_fn(x)
    # we turn the vector again into a state_dict
    grads_ = x_to_params(x_, sizes)
    # and we use the state dict to update the gradients
    for n, p in model.named_parameters():
        if is_shared(n):
            try:
                p.grad = grads_[n].clone()
            except:
                continue


"""
    Updates the gradients of the model - but layer-wise
    :param nn.Module model: the neural network
    :param function grad_fn: function to be applied to the gradient of the layers
"""
def update_gradients_layerwise(model, grad_fn):
    for k, (m, params) in enumerate(model.named_parameters()):
        params.grad.data = grad_fn(params.grad.data, k)


"""
  freeze the given layers of the model
  :param nn.Module model: neural network
  :param list layers: list of layers to be frozen
"""
def freeze(model, layers):
    for n, p in model.named_parameters():
        if n in layers:
            p.grad = torch.zeros_like(p)


"""
   train the neural network on the given task with the given settings
   :param nn.Module net_: the neural network
   :param int task: task id of the task to learn
   :param function reg_loss: (optional) function to compute the regularization loss (e.g. for EWC, MAS, etc.)
   :param function grad_fn: (optional) function to be applied to the gradients (e.g. for OGD)
   :param function lay_grad_fn: (optional) function to be applied to the gradients, but layer-wise (e.g. for GPM)
   :param int epochs: (optional) number of epochs to train, default is num_epochs passed to init 
   :param bool eval: (optional) if True, the neural net is evaluated on dev set after each epoch
   :param list freeze_layers: (optional) layers to freeze
   :param str opt: (optional) optimizer to learn the network with (adam or sgd) - default is adam
"""
def train_net(net_, task, reg_loss=None, grad_fn=None, lay_grad_fn=None, epochs=-1, eval=False, save='',
              freeze_layers=[], opt='adam'):
    if epochs == -1:
        epochs = n_epochs
    criterion = nn.CrossEntropyLoss()
    if opt == 'sgd':
        optimizer = optim.SGD(net_.parameters(), lr=0.001, momentum=0.9)
    else:
        optimizer = optim.Adam(net_.parameters())
    k = 0
    trainloader_ = train_data(task)
    net_ = net_.to(device)
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader_, 0):
            optimizer.zero_grad()
            inputs, labels = data
            if n_classes is not None:
                labels = labels % n_classes
            outputs = net_(inputs.to(device)) if not multihead else net_(inputs.to(device), task)
            loss = criterion(outputs, labels.to(device))
            if reg_loss is not None:
                try:
                    loss += reg_loss(net_)
                except:
                    loss += reg_loss(net_, inputs)
            loss.backward()
            if grad_fn is not None:
                update_gradients(net_, grad_fn)
            if lay_grad_fn is not None:
                update_gradients_layerwise(net_, lay_grad_fn)
            if len(freeze_layers) > 0:
                freeze(net_, freeze_layers)
            optimizer.step()
            running_loss += loss.item()
            k += 1
        if not eval:
            logger.debug('[%d, %5d] loss: %.3f' % (epoch + 1, k + 1, running_loss / k))
        else:
            acc = test(net_, task, dataset='dev', print_result=False)
            logger.debug('[%d, %5d] loss: %.3f --> acc on dev set: %.2f' % (epoch + 1, k + 1, running_loss / k, acc))
            net_.to(device)
        if len(save) > 0:
            torch.save(net_.state_dict(), save + "_ep.%d" % (epoch+1))
        running_loss, k = 0, 0
    del trainloader_
    net_.cpu()
    logger.debug('Finished Training')
