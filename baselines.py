import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import logging
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

""" 
    Baselines implementations: 
       -- Elastic Weight Consolidation (EWC) (Kirkpatrick et al., 2016)
       -- Memory-Aware Synapses (MAS) (Aljundi et al., 2018)
       -- Learning Without Forgetting (LWF) (Li et al., 2017)
       -- Orthogonal Gradient Descent (OGD) (Farajtabar et al., 2019)
       -- Kronecker-Factored Laplace Approximation (KF) (Ritter et al., 2018) 
       
    References:
    James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, 
        John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, 
        and Raia Hadsell. Overcoming catastrophic forgetting in neural networks, 2017.
    Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny, Marcus Rohrbach, and Tinne Tuytelaars. 
        Memory aware synapses: Learning what (not) to forget, 2018.
    Zhizhong Li and Derek Hoiem. Learning without forgetting, 2017
    Mehrdad Farajtabar, Navid Azizan, Alex Mott, and Ang Li. Orthogonal gradient descent for
        continual learning, 2019.
    Hippolyt Ritter, Aleksandar Botev, and David Barber. 2018. Online structured Laplace Approximations for Overcoming 
        Catastrophic Forgetting. In Proceedings of the 32nd International Conference on Neural Information Processing 
        Systems (NIPS'18). Curran Associates Inc., Red Hook, NY, USA, 3742-3752.
"""

logger = logging.getLogger('main')
logger.debug('Device in baselines.py = %s' % str(device))


"""
    Elastic Weight Consolidation (Kirkpatrick et al., 2016)
    :param list shared_layers: (optional) if provided, only layers in shared_layers are subject to regularization
    :param int n_classes: (optional) required for split experiments
"""
class EWC:
    def __init__(self, shared_layers=None, n_classes=None):
        self.is_shared = lambda x: shared_layers is None or x in shared_layers
        self.importance_weights = None
        self.old_net = None
        self.n_classes = n_classes

    """ 
        Computes the diagonal of the Fisher Information
        :param nn.Module net_: the neural network
        :param torch.Dataloader dataset: the data
        :param int task: task id, only required if multi-head classification
     """
    def compute_FIM(self, net_, dataset, task=None):
        iw = {n: torch.zeros_like(p, device=device) for n, p in net_.named_parameters() if self.is_shared(n)}
        criterion = nn.CrossEntropyLoss()
        net_ = net_.to(device)
        k = 0
        for i, data in enumerate(dataset, 0):
            net_.zero_grad()
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # we map classes to [0 - 9]
            if self.n_classes is not None:
                labels = labels % self.n_classes
            # forward + backward + optimize
            outputs = net_(inputs.to(device), task) if task is not None else net_(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            for n, p in net_.named_parameters():
                if self.is_shared(n):
                    iw[n] = iw[n] + p.grad ** 2
            k += 1
        self.old_net = {n: copy.deepcopy(p.detach()) for n, p in net_.named_parameters() if self.is_shared(n)}
        if self.importance_weights is None:
            self.importance_weights = {n: p / k for n, p in iw.items()}
        else:
            self.importance_weights = {n: self.importance_weights[n] + p / k for n, p in iw.items()}
        logger.debug("Norm of FIM = %s" % str(torch.norm(torch.cat([p.view(-1) for n, p in
                                                                    self.importance_weights.items()]))))
        net_.cpu()

    """ 
       Returns the regularization loss for EWC 
       :param nn.Module current_net: the network subject to regularization
       :param float alpha: the weight of the regularization
    """
    def regularize(self, current_net, alpha):
        loss = 0
        for n, p in current_net.named_parameters():
            if self.is_shared(n):
                _loss = self.importance_weights[n] * (p - self.old_net[n]) ** 2
                loss += _loss.sum()
        return alpha / 2 * loss


"""
    Memory-Aware Synapses (Aljundi et al., 2018)
    :param list shared_layers: (optional) if provided, it contains the layers subject to regularization
"""
class MAS:
    def __init__(self, shared_layers=None):
        self.is_shared = lambda x: shared_layers is None or x in shared_layers
        self.old_net = None
        self.importance_weights = None

    """ 
       Computes the importance weights for MAS 
       :param nn.Module net_: the neural network
       :param torch.Dataloader: the dataset
       :param int task: (optional) task id required for multi-head classification
    """
    def compute_IW(self, net_, dataset, task=None):
        iw = {n: torch.zeros_like(p, device=device) for n, p in net_.named_parameters() if self.is_shared(n)}
        net_ = net_.to(device)
        sm = nn.Softmax(dim=1)
        k = 0
        for i, data in enumerate(dataset, 0):
            net_.zero_grad()
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # forward + backward + optimize
            outputs = net_(inputs.to(device), task) if task is not None else net_(inputs.to(device))
            loss = torch.norm(sm(outputs)) ** 2
            loss.backward()
            for n, p in net_.named_parameters():
                if self.is_shared(n):
                    iw[n] = iw[n] + abs(p.grad)
            k += 1
        self.old_net = {n: copy.deepcopy(p.detach()) for n, p in net_.named_parameters() if self.is_shared(n)}
        if self.importance_weights is None:
            self.importance_weights = {n: p / k for n, p in iw.items()}
        else:
            self.importance_weights = {n: self.importance_weights[n] + p / k for n, p in iw.items()}
        logger.debug("Norm of IW = %s" % str(torch.norm(torch.cat([p.view(-1) for n, p in
                                                                   self.importance_weights.items()]))))
        net_.cpu()

    """ 
       Returns the regularization loss for MAS 
       :param nn.Module current_net: the neural network subject to regularization
       :param float alpha: the weight of the regularization
    """
    def regularize(self, current_net, alpha):
        loss_ = 0
        for n, p in current_net.named_parameters():
            if self.is_shared(n):
                _loss = self.importance_weights[n] * (p - self.old_net[n]) ** 2
                loss_ += _loss.sum()
        return alpha / 2 * loss_


"""
    Learning without Forgetting (Li et al., 2017)
    :param float temperature: (optional) temperature for knowledge distillation, default 2 (as in the original paper)
"""
class LWF:
    def __init__(self,  temperature=2):
        self.T = temperature

    """
        Knowledge distillation between teacher and student output
        :param torch.tensor teacher: teacher output
        :param torch.tensor student: student output
    """
    def knowledge_distillation(self, teacher, student):
        sm, lsm = torch.nn.Softmax(dim=1), torch.nn.LogSoftmax(dim=1)
        prob_t = sm(teacher / self.T)
        log_s = lsm(student / self.T)
        return -(prob_t * log_s).sum()

    """
        Updates the teacher network (used as teacher in knowledge distillation)
        :param nn.Module net_: the teacher neural network 
    """
    def set_old_net(self, net_):
        self.old_net = copy.deepcopy(net_).to(device)

    """
        Computes the LWF regularization loss
        :param nn.Module current_net: the network subject to regularization
        :param torch.tensor inputs: the input batch
        :param float alpha: weight of the the regularization loss
        :param int task: (optional) required if multi-head classification
    """
    def regularize(self, current_net, inputs, alpha, task=None):
        loss = 0
        if task is not None:
            for i in range(task):
                student_output = current_net(inputs.to(device), i)
                teacher_outputs = self.old_net(inputs.to(device), i).detach()
                loss += self.knowledge_distillation(teacher_outputs, student_output)
        else:
            student_output = current_net(inputs.to(device))
            teacher_outputs = self.old_net(inputs.to(device)).detach()
            loss += self.knowledge_distillation(teacher_outputs, student_output)

        return alpha * loss


"""
    Orthogonal Gradient Descent (Farajtabar et al., 2019)
    :param list shared_layers: (optional) if provided, the layers subject to regularization
    :param int M: (optional) number of gradients to store per task - default 200 as in the original paper
    :param int n_classes: (optional) required if multi-head classification 
"""
class OGD:
    def __init__(self, shared_layers=None, M=200, n_classes=None):
        self.is_shared = lambda x: shared_layers is None or x in shared_layers
        self.S = None
        self.n_classes = n_classes
        self.M = M

    """
        Transform dictionary of tensors into 1D tensor
        :param dict param: dictionary of tensors
    """
    @staticmethod
    def to_vec(param):
        return torch.cat([p.view(-1) for n, p in param.items()])

    """
        Orthogonalize using Gram-Schmidt, such that vec is orthogonal to all columns of S
        :param torch.tensor vec: the tensor (1D) of size N to orthogonalize
        :param torch.tensor S: 2D tensor with size N*K
    """
    @staticmethod
    def gram_schmidt(vec, S):
        for i in range(S.size(1)):
            vec = vec - torch.dot(vec, S[:, i]) / torch.dot(S[:, i], S[:, i]) * S[:, i]
        return vec

    """
        Compute the gradients for the current task
        :param nn.Module net_: the neural network
        :param torch.Dataloader data: the data
        :param int task: (optional) the task id, required if multi-head classification
    """
    def compute_gradients(self, net_, data, task=None):
        net_.to(device)
        k, S = 0, None
        for i, data in enumerate(data, 0):
            net_.zero_grad()
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if self.n_classes is not None:
                labels = labels % self.n_classes
            # forward + backward + optimize
            outputs = net_(inputs.to(device), task) if task is not None else net_(inputs.to(device))
            loss = outputs[0, labels[0]]
            loss.backward()
            iw = {}
            for n, p in net_.named_parameters():
                if self.is_shared(n):
                    iw[n] = 1.0 * p.grad
            s = self.to_vec(iw)
            try:
                s = self.gram_schmidt(s, torch.transpose(S, 0, 1))
                S = torch.cat((S, s.view(1, -1)), 0)
            except:
                S = s.view(1, -1)
            k += 1
            if k >= self.M:
                break
        S = torch.transpose(S, 0, 1)
        if self.S is None:
            self.S = S.to(device)
        else:
            self.S = torch.cat((self.S, S), dim=1).to(device)
        logger.debug("OGD: S has size %s" % str(self.S.size()))
        net_.cpu()

    """
        OGD regularization: must be applied to gradient
        :param torch.tensor grad: gradient of neural network as 1D tensor
    """
    def regularize(self, grad):
        return self.gram_schmidt(grad, self.S)


def cuda_overview():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        t = torch.cuda.get_device_properties(0).total_memory * 10.0 ** (-9)
        r = torch.cuda.memory_reserved(0) * 10.0 ** (-9)
        a = torch.cuda.memory_allocated(0) * 10.0 ** (-9)
        f = r - a  # free inside reserved
        logger.debug("CUDA memory - total: %.4f; reserved: %.4f; allocated: %.4f; available: %.4f" % (t, r, a, f))
    else:
        pass

"""
    Kronecker-factored Laplace Approximations (Ritter et al., 2018)
"""
class KF:
    def __init__(self):
        self.Q = {}
        self.H = {}
        self.task = -1

    """
        Transform dictionary of tensors into 1D tensor
        :param dict param: dictionary of tensors
    """
    @staticmethod
    def to_vec(param):
        return torch.cat([p.view(-1) for n, p in param.items()])

    """
        Update Q and H matrices - called when consolidating knowledge from a new task
        :param nn.Module net_: the neural network
        :param torch.Dataloader dataloader: dataloader when representative data of current task
        :param int task: (optional) should be provided in case of multi-head classification
    """
    def update_KF(self, net_, dataloader, task=None):
        logger.debug('Updating KF...')
        net_ = net_.to(device)
        criterion = nn.CrossEntropyLoss()
        Q, H, k = {}, {}, 0
        for i, data in enumerate(dataloader, 0):
            net_.zero_grad()
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            outputs = net_(inputs.to(device), task) if task is not None else net_(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            Q, H = self.compute_HQ(net_, loss, Q, H)
            torch.cuda.empty_cache()
            k += 1
        self.task += 1
        self.Q[self.task], self.H[self.task] = {n: p / k for n, p in Q.items()}, {n: p / k for n, p in H.items()}
        self.old_net = {n: copy.deepcopy(p.detach()) for n, p in net_.named_parameters()}
        net_.cpu()

    """
        Help function to update Q and H in self.update_KF
        :param nn.Module net_: the neural network
        :param torch.tensor loss: the loss as computed on the current mini-batch
        :param dict Q: dictionary with the Q matrices
        :param dict H: dictionary with the H matrices
    """
    def compute_HQ(self, net_, loss, Q, H):
        for layer in net_.layers_in_reverse_order:
            if 'conv' in layer:
                q, h = self.compute_HQ_conv(net_, layer, loss)
            elif 'bn' in layer:
                #q, h = self.compute_HQ_bn(net_, layer, loss)
                continue
            else:
                q, h = self.compute_HQ_lin(net_, layer, loss)
            try:
                Q[layer], H[layer] = Q[layer] + q.detach(), H[layer] + h.detach()
            except Exception as e:
                logger.debug('Exception: %s' % e)
                Q[layer], H[layer] = q.detach(), h.detach()
        return Q, H

    """
        Help function to compute Q and H for a FC layer
        :param nn.Module net_: the neural network
        :param str layer: name or identifier of layer
        :param torch.tensor loss: the loss on the current mini-batch
    """
    def compute_HQ_lin(self, net_, layer, loss):
        logger.debug("Size of %s = %s" % (layer, str(net_.kf['input'][layer].size())))
        OUT, IN = net_.kf['input'][layer].size()
        A = torch.cat((net_.kf['input'][layer].detach(), torch.ones([OUT], device=device).view(OUT, 1)), dim=1)
        #A = net_.kf['input'][layer]
        Q = (1 / A.size(0)) * torch.matmul(torch.transpose(A, 0, 1), A)
        g = torch.autograd.grad(loss, net_.kf['pre-activation'][layer], create_graph=True)[0].detach()
        H = (1 / g.size(0)) * torch.matmul(torch.transpose(g, 0, 1), g)
        return Q.detach(), H.detach()

    """
        Help function to compute Q and H for a Convolutional layer
        :param nn.Module net_: the neural network
        :param str layer: name or identifier of layer
        :param torch.tensor loss: the loss on the current mini-batch
    """
    def compute_HQ_conv(self, net_, layer, loss):
        I, J, D, D = net_.state_dict()[layer].size()
        M, _, T, _ = net_.kf['input'][layer].size()
        _, _, Tn, _ = net_.kf['pre-activation'][layer].size()
        exp_A = self.expansion_operator(net_.kf['input'][layer].detach(), M, J, T, D)
        exp_A = torch.cat((exp_A, torch.ones([M * T * T]).view(-1, 1)), dim=1)
        Q = 1.0 / M * torch.matmul(torch.transpose(exp_A, 0, 1), exp_A).to(device)
        dS = torch.autograd.grad(loss, net_.kf['pre-activation'][layer], create_graph=True)[0].detach()
        S = torch.transpose(dS, 0, 1).reshape(I, M * Tn * Tn)
        H = 1 / (Tn * Tn * M) * torch.matmul(S, torch.transpose(S, 0, 1))
        return Q.detach(), H.detach()

    """
        The expansion operator as explained by (Gross and Martens, 2016)
        
        References:
        R. Grosse and J. Martens. A Kronecker-factored Approximate Fisher Matrix for Convolution Layers. 
              In International Conference on Machine Learning, pages 573-582, 2016.
    """
    @staticmethod
    def expansion_operator(A, M, J, T, D):
        d = D // 2
        B = F.pad(A, (d, d, d, d))
        C = torch.zeros([M * T * T, J * D * D])
        for m in range(M):
            for j in range(J):
                for t1 in range(T):
                    for t2 in range(T):
                        t = t1 * T + t2
                        C[t * M + m, j * D * D:(j + 1) * D * D] = B[m, j, t1: t1 + d + d + 1,
                                                                  t2: t2 + d + d + 1].reshape(-1)
        return C

    """
        To regularize the training of the network
        :param nn.Module current_net: the current neural network 
        :param float alpha: the weight of the regularization
    """
    def regularize(self, current_net, alpha):
        loss = 0
        for n, p in current_net.named_parameters():
            if n in current_net.layers_in_reverse_order:
                if 'conv' in n:
                    I, J, D, _ = p.size()
                    param, old_param = p.view(I, J * D * D), self.old_net[n].reshape(I, J * D * D)
                else:
                    param, old_param = 1.0 * p, self.old_net[n]
                layer_name = n.split('.')[0]
                for nb, pb in current_net.named_parameters():
                    if nb == layer_name + ".bias":
                        param, old_param = torch.cat((param, pb.view(-1, 1)), dim=1), \
                                           torch.cat((old_param, self.old_net[layer_name + ".bias"].view(-1, 1)), dim=1)
                        break
                for task in range(self.task + 1):
                    logger.debug("Size of: H = %s, param = %s, old_param = %s, Q = %s" %
                                 (str(self.H[task][n].size()), str(param.size()), str(old_param.size()),
                                  str(self.Q[task][n].size())))
                    HWQ = torch.matmul(self.H[task][n], torch.matmul(param - old_param, self.Q[task][n]))
                    loss += alpha / 2 * torch.dot((param - old_param).view(-1), HWQ.view(-1))
        return loss
