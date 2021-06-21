import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import logging
import numpy as np
import os
import os.path
import urllib.request
from PIL import Image
import pickle
from torchvision.datasets.mnist import *


logger = logging.getLogger('main')



"""
    Returns train, validation and test dataloaders for the Rotated MNIST experimenets
    :param int n_tasks: the number of tasks to create
    :param int angle: the angle by which the images for two subsequent tasks are rotated
    :param int batch_size: the batch_size
    ::param bool return_joint: (optional) if True, a dataloader for the joint dataset is also returned
"""


def get_rotated_mnist_data(n_tasks, angle, batch_size, return_joint=False):
    logger.debug("Rotated MNIST dataset with: angle = %d, n_tasks = %d, batch_size = %d" % (angle, n_tasks, batch_size))
    rot_transforms = {i: transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
                                             transforms.RandomRotation([angle * i, angle * i])]) for i in range(n_tasks)}

    dataset = {i: torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=rot_transforms[i]) for i in range(n_tasks)}
    testset = {i: torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=rot_transforms[i]) for i in range(n_tasks)}

    # Split test in dev and test:
    trainset, devset = {}, {}
    for n, p in dataset.items():
        trainset[n], devset[n] = torch.utils.data.random_split(p, [55000, 5000], generator=torch.Generator().manual_seed(42))

    joint_trainset = torch.utils.data.ConcatDataset([trainset[i] for i in trainset.keys()])
    train_data = lambda t_: torch.utils.data.DataLoader(trainset[t_], batch_size=batch_size, shuffle=True, num_workers=0)
    test_data = lambda t_: torch.utils.data.DataLoader(testset[t_], batch_size=batch_size, shuffle=False, num_workers=0)
    dev_data = lambda t_: torch.utils.data.DataLoader(devset[t_], batch_size=batch_size, shuffle=False, num_workers=0)

    if not return_joint:
        return train_data, dev_data, test_data
    else:
        return train_data, dev_data, test_data, joint_trainset


"""
    Returns train, validation and test dataloaders for the Split MNIST experiments
    :param int n_tasks: the number of tasks to create
    :param int batch_size: the batch_size
"""


def get_split_mnist_data(n_tasks, batch_size):
    logger.debug("Split MNIST dataset with: n_tasks = %d, batch_size = %d" % (n_tasks, batch_size))
    n_classes = 10 // n_tasks
    classes_per_task = {i: [i * n_classes + j for j in range(n_classes)] for i in range(n_tasks)}
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize( (0.1307,), (0.3081,))])

    def get_indices(data, class_names, datas=None):
        indices, temp = [], []
        if datas is not None:
            for i in range(len(data.indices)):
                if datas.targets[data.indices[i]] in class_names:
                    indices.append(i)
        else:
            for i in range(len(data.targets)):
                if data.targets[i] in class_names:
                    indices.append(i)

        return indices

    # loading the data
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # loading the label names
    classes = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)

    def get_classes(taskid):
        cls = classes_per_task[taskid]
        name = ''
        for iteration in range(len(cls)):
            name += str(classes[cls[iteration]])
            if iteration != len(cls) - 1:
                name += ', '
        return name

    for i in range(n_tasks):
        logger.debug('Task %d consists of: %s' % (i, str(get_classes(i))))

    # Split test in dev and test:
    trainset, devset = torch.utils.data.random_split(dataset, [55000, 5000], generator=torch.Generator().manual_seed(42))
    train_data = lambda t_: torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=0,
                            sampler=torch.utils.data.sampler.SubsetRandomSampler(get_indices(trainset, classes_per_task[t_], dataset)))
    test_data = lambda t_: torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=0,
                            sampler=torch.utils.data.sampler.SubsetRandomSampler(get_indices(testset, classes_per_task[t_])))
    dev_data = lambda t_: torch.utils.data.DataLoader(devset, batch_size=batch_size, num_workers=0,
                            sampler=torch.utils.data.sampler.SubsetRandomSampler(get_indices(devset, classes_per_task[t_], dataset)))
    return train_data, dev_data, test_data, n_classes


"""
    Returns train, validation and test dataloaders for the Split CIFAR-100 experiments
    :param int n_tasks: the number of tasks to create
    :param int batch_size: the batch_size
"""


def get_split_cifar100_data(n_tasks, batch_size):
    logger.debug("Split CIFAR-100 dataset with: n_tasks = %d, batch_size = %d" % (n_tasks, batch_size))
    n_classes = 100 // n_tasks
    classes_per_task = {i: [i * n_classes + j for j in range(n_classes)] for i in range(n_tasks)}
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def get_indices(data, class_names, datas=None):
        indices, temp = [], []
        if datas is not None:
            for i in range(len(data.indices)):
                if datas.targets[data.indices[i]] in class_names:
                    indices.append(i)
        else:
            for i in range(len(data.targets)):
                if data.targets[i] in class_names:
                    indices.append(i)

        return indices
    # loading the data
    dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    # loading the label names
    import pickle
    pickleFile = open("data/cifar-100-python/meta", 'rb')
    classes = pickle.load(pickleFile)
    classes = classes['fine_label_names']
    pickleFile.close()

    def get_classes(taskid):
        cls = classes_per_task[taskid]
        name = ''
        for iteration in range(len(cls)):
            name += classes[cls[iteration]]
            if iteration != len(cls) - 1:
                name += ', '
        return name

    for i in range(n_tasks):
        logger.debug('Task %d consists of: %s' % (i, str(get_classes(i))))

    # Split test in dev and test:
    trainset, devset = torch.utils.data.random_split(dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))
    train_data = lambda t_: torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=0,
                            sampler=torch.utils.data.sampler.SubsetRandomSampler(get_indices(trainset, classes_per_task[t_], dataset)))
    test_data = lambda t_: torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=0,
                            sampler=torch.utils.data.sampler.SubsetRandomSampler(get_indices(testset, classes_per_task[t_])))
    dev_data = lambda t_: torch.utils.data.DataLoader(devset, batch_size=batch_size, num_workers=0,
                            sampler=torch.utils.data.sampler.SubsetRandomSampler(get_indices(devset, classes_per_task[t_], dataset)))
    return train_data, dev_data, test_data, n_classes

"""
    Returns train, validation and test dataloaders for the Split CIFAR-10/100 experiments
    :param int n_tasks: the number of tasks to create for the CIFAR-100 (total tasks is thus n_tasks + 1)
    :param int batch_size: the batch_size
"""

def get_split_cifar10_100_data(n_tasks, batch_size):
    logger.debug("Split CIFAR-10/100 dataset with: n_tasks = %d, batch_size = %d" % (n_tasks+1, batch_size))
    n_classes = 100 // n_tasks
    classes_per_task = {i: [i * n_classes + j for j in range(n_classes)] for i in range(n_tasks)}

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def get_indices(data, class_names, datas=None):
        indices, temp = [], []
        if datas is not None:
            for i in range(len(data.indices)):
                if datas.targets[data.indices[i]] in class_names:
                    indices.append(i)
        else:
            for i in range(len(data.targets)):
                if data.targets[i] in class_names:
                    indices.append(i)
        return indices
    # load the cifar10 dataset
    dataset10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset10 = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    classes10 = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # load the cifar100 dataset
    dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    # load the class labels
    import pickle
    pickleFile = open("data/cifar-100-python/meta", 'rb')
    classes = pickle.load(pickleFile)
    classes = classes['fine_label_names']
    pickleFile.close()

    def get_classes(taskid):
        cls = classes_per_task[taskid - 1] if taskid > 0 else [i for i in range(len(classes10))]
        name = ''
        for iteration in range(len(cls)):
            name += str(classes[cls[iteration]]) if taskid > 0 else str(classes10[cls[iteration]])
            if iteration != len(cls) - 1:
                name += ', '
        return name

    for i in range(n_tasks+1):
        logger.debug('Task %d consists of: %s' % (i, get_classes(i)))

    # Split test in dev and test:
    trainset10, devset10 = torch.utils.data.random_split(dataset10, [45000, 5000], generator=torch.Generator().manual_seed(42))
    trainset, devset = torch.utils.data.random_split(dataset, [45000, 5000], generator=torch.Generator().manual_seed(42))

    train_data100 = lambda t_: torch.utils.data.DataLoader(trainset, batch_size=batch_size, num_workers=0,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(get_indices(trainset, classes_per_task[t_], dataset)))
    test_data100 = lambda t_: torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=0,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(get_indices(testset, classes_per_task[t_])))
    dev_data100 = lambda t_: torch.utils.data.DataLoader(devset, batch_size=batch_size, num_workers=0,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(get_indices(devset, classes_per_task[t_], dataset)))
    train_data10 = torch.utils.data.DataLoader(trainset10, batch_size=batch_size, num_workers=0, shuffle=True)
    test_data10 = torch.utils.data.DataLoader(testset10, batch_size=batch_size, num_workers=0, shuffle=False)
    dev_data10 = torch.utils.data.DataLoader(devset10, batch_size=batch_size, num_workers=0, shuffle=False)

    train_data = lambda t_: train_data10 if t_ == 0 else train_data100(t_ - 1)
    dev_data = lambda t_: dev_data10 if t_ == 0 else dev_data100(t_ - 1)
    test_data = lambda t_: test_data10 if t_ == 0 else test_data100(t_ - 1)

    return train_data, dev_data, test_data, n_classes



"""
    Generates a MNIST permutation dataset
    :param np.RandomState random_state: numpy random state object 
"""
def generate_mnist_permutation(random_state):
  idx_permute = torch.from_numpy(random_state.permutation(28*28))
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
                                  transforms.Lambda(lambda x: x.view(-1)[idx_permute].view(1, 28, 28))])
  dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
  testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
  return dataset, testset


"""
    Returns train, validation and test dataloaders for the Permuted MNIST experiments
    :param int n_tasks: the number of tasks to create
    :param int batch_size: the batch_size
    :param bool return_joint: (optional) joint dataset is returned if set to True
"""

def get_permuted_mnist_data(n_tasks, batch_size, return_joint=False):
    logger.debug("Permuted MNIST dataset with: n_tasks = %d, batch_size = %d" % (n_tasks, batch_size))
    rng_permute = np.random.RandomState(92916)

    dataset, testset = {}, {}
    for i in range(n_tasks):
        dataset[i], testset[i] = generate_mnist_permutation(rng_permute)

    # Split test in dev and test:
    trainset, devset = {}, {}
    for n, p in dataset.items():
        trainset[n], devset[n] = torch.utils.data.random_split(p, [55000, 5000], generator=torch.Generator().manual_seed(42))

    joint_trainset = torch.utils.data.ConcatDataset([trainset[i] for i in trainset.keys()])
    train_data = lambda t_: torch.utils.data.DataLoader(trainset[t_], batch_size=batch_size, shuffle=True, num_workers=0)
    test_data = lambda t_: torch.utils.data.DataLoader(testset[t_], batch_size=batch_size, shuffle=False, num_workers=0)
    dev_data = lambda t_: torch.utils.data.DataLoader(devset[t_], batch_size=batch_size, shuffle=False, num_workers=0)

    if not return_joint:
        return train_data, dev_data, test_data
    else:
        return train_data, dev_data, test_data, joint_trainset




class notMNIST(torchvision.datasets.MNIST):
    """`notMNIST <https://github.com/davidflanagan/notMNIST-to-MNIST>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/processed/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    training_file = 'training.pt'
    test_file = 'test.pt'
    classes = ['0 - A', '1 - B', '2 - C', '3 - D', '4 - E',
               '5 - F', '6 - G', '7 - H', '8 - I', '9 - J']

    #def __init__(self, root, train, transform=None, target_transform=None, download=False):
    #    super(notMNIST, self).__init__(root, train, transform=transform,
    #                                   target_transform=target_transform, download=download)

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""

        if self._check_exists():
            return

        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

"""
    Returns train, validation and test dataloaders for the 'Vision Datasets' experiments
    :param int batch_size: the batch_size
    :param int Nsamples: number of samples drawn from each training set to compute FIM, Hessian etc. 
    :param bool return_joint: (optional) joint dataset is returned if set to True
"""

def get_five_datasets(batch_size, Nsamples=50, return_joint=False):
    logger.debug("Five datasets with batch_size = %d" % (batch_size))

    dataset, testset = {}, {}
    tr_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),
                    transforms.Resize(32), transforms.CenterCrop(32), transforms.Lambda(lambda x: x.repeat(3, 1, 1))])
    dataset[0] = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=tr_mnist)
    testset[0] = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=tr_mnist)
    tr_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                std=[0.229, 0.224, 0.225])])
    dataset[1] = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=tr_cifar)
    testset[1] = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=tr_cifar)
    tr_svhn = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.4377,0.4438,0.4728],
                                                                                std=[0.198,0.201,0.197])])
    dataset[2] = torchvision.datasets.SVHN(root='./data', split='train', download=True, transform=tr_svhn)
    testset[2] = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=tr_svhn)
    tr_fmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2190,), (0.3318,)),
                    transforms.Resize(32), transforms.CenterCrop(32), transforms.Lambda(lambda x: x.repeat(3, 1, 1))])
    dataset[3] = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=tr_fmnist)
    testset[3] = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=tr_fmnist)
    tr_nmnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4254,), (0.4501,)),
                    transforms.Resize(32), transforms.CenterCrop(32), transforms.Lambda(lambda x: x.repeat(3, 1, 1))])
    dataset[4] = notMNIST(root="./data", train=True, download=True, transform=tr_nmnist)
    testset[4] = notMNIST(root='./data', train=False, download=True, transform=tr_nmnist)

    sizes = {0: 60000, 1: 50000, 2: 73257, 3: 60000, 4: len(dataset[4])}
    names = {0: 'MNIST', 1: 'CIFAR-10', 2: 'SVHN', 3: 'FashionMNIST', 4: 'notMNIST'}

    # Split test in dev and test:
    trainset, devset, sampleset = {}, {}, {}
    for n, p in dataset.items():
        train = int(0.95 * sizes[n])
        trainset[n], devset[n] = torch.utils.data.random_split(p, [train, sizes[n] - train], generator=torch.Generator().manual_seed(42))
        logger.debug("For dataset = %s: [train = %s, dev = %s]" % (names[n], str(train), str(sizes[n] - train)))
        sampleset[n] = torch.utils.data.Subset(trainset[n], np.random.permutation(len(trainset[n]))[:Nsamples*batch_size])

    joint_trainset = torch.utils.data.ConcatDataset([trainset[i] for i in trainset.keys()])
    train_data = lambda t_: torch.utils.data.DataLoader(trainset[t_], batch_size=batch_size, shuffle=True, num_workers=0)
    test_data = lambda t_: torch.utils.data.DataLoader(testset[t_], batch_size=batch_size, shuffle=False, num_workers=0)
    dev_data = lambda t_: torch.utils.data.DataLoader(devset[t_], batch_size=batch_size, shuffle=False, num_workers=0)
    sample_data = lambda t_: torch.utils.data.DataLoader(sampleset[t_], batch_size=batch_size, shuffle=True, num_workers=0)

    if not return_joint:
        return train_data, dev_data, test_data, sample_data
    else:
        return train_data, dev_data, test_data, joint_trainset



