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
       -- Gradient Projection Memory (GPM) (Saha et al., 2021) 
       
    References:
    James Kirkpatrick, Razvan Pascanu, Neil Rabinowitz, Joel Veness, Guillaume Desjardins, Andrei A. Rusu, Kieran Milan, 
        John Quan, Tiago Ramalho, Agnieszka Grabska-Barwinska, Demis Hassabis, Claudia Clopath, Dharshan Kumaran, 
        and Raia Hadsell. Overcoming catastrophic forgetting in neural networks, 2017.
    Rahaf Aljundi, Francesca Babiloni, Mohamed Elhoseiny, Marcus Rohrbach, and Tinne Tuytelaars. 
        Memory aware synapses: Learning what (not) to forget, 2018.
    Zhizhong Li and Derek Hoiem. Learning without forgetting, 2017
    Mehrdad Farajtabar, Navid Azizan, Alex Mott, and Ang Li. Orthogonal gradient descent for
        continual learning, 2019.
    Gobinda Saha, Isha Garg, and Kaushik Roy. Gradient projection memory for continual learning, 2021.
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


"""
    Gradient Projection Memory (Saha et al., 2021) 
    - implementation adapted from 'https://github.com/sahagobinda/GPM'
    :param np.array threshold: threshold per layer 
    :param int n_samples: number of samples to take for each task and layer
"""
class GPM:
    def __init__(self, threshold=np.array([0.95, 0.99, 0.99]), n_samples=300):
        self.threshold = threshold
        self.n_samples = n_samples

    """
        Computes the representation matrix
        :param nn.Module net: the neural network
        :param torch.Dataloader dataloader: the training data to take samples from
    """
    def get_representation_matrix(self, net, dataloader):
        # Collect activations by forward pass
        r = np.arange(len(dataloader))
        np.random.shuffle(r)
        r = r[0:self.n_samples]
        b = torch.tensor([])
        for i, data in enumerate(dataloader):
            x, y = data
            if i in r:
                b = torch.cat((b, x.view(-1, 28 * 28)), dim=0)
        example_data = b.to(device)
        _ = net(example_data)

        batch_list = [self.n_samples, self.n_samples, self.n_samples]
        mat_list = []  # list contains representation matrix of each layer
        act_key = list(net.act.keys())

        for i in range(len(act_key)):
            bsz = batch_list[i]
            act = net.act[act_key[i]].detach().cpu().numpy()
            activation = act[0:bsz].transpose()
            mat_list.append(activation)
        return mat_list

    """
        Updates the GPM
        :param list mat_list: representation matrix as a list of np.arrays
        :param feature_list: (optional) GPM from previous tasks (if applicable)
    """
    def update_GPM(self, mat_list, feature_list=None):
        if feature_list is None:
            # After First Task
            feature_list = []
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U, S, Vh = np.linalg.svd(activation, full_matrices=False)
                # criteria (Eq-5)
                sval_total = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                r = np.sum(np.cumsum(sval_ratio) < self.threshold[i])  # +1
                feature_list.append(U[:, 0:r])
        else:
            for i in range(len(mat_list)):
                activation = mat_list[i]
                U1, S1, Vh1 = np.linalg.svd(activation, full_matrices=False)
                sval_total = (S1 ** 2).sum()
                # Projected Representation (Eq-8)
                act_hat = activation - np.dot(np.dot(feature_list[i], feature_list[i].transpose()), activation)
                U, S, Vh = np.linalg.svd(act_hat, full_matrices=False)
                # criteria (Eq-9)
                sval_hat = (S ** 2).sum()
                sval_ratio = (S ** 2) / sval_total
                accumulated_sval = (sval_total - sval_hat) / sval_total

                r = 0
                for ii in range(sval_ratio.shape[0]):
                    if accumulated_sval < self.threshold[i]:
                        accumulated_sval += sval_ratio[ii]
                        r += 1
                    else:
                        break
                if r == 0:
                    print ('Skip Updating GPM for layer: {}'.format(i + 1))
                    continue
                # update GPM
                Ui = np.hstack((feature_list[i], U[:, 0:r]))
                if Ui.shape[1] > Ui.shape[0]:
                    feature_list[i] = Ui[:, 0:Ui.shape[0]]
                else:
                    feature_list[i] = Ui
        return feature_list

    """
        Computes the feature matrix, used for the regularization
        :param nn.Module model: the neural network
        :param list feature_list: representing the GPM
    """
    @staticmethod
    def get_feature_mat(model, feature_list):
        feature_mat = []
        for i in range(len(model.act)):
            Uf = torch.Tensor(np.dot(feature_list[i], feature_list[i].transpose())).to(device)
            feature_mat.append(Uf)
        return feature_mat

    """
        Updates the GPM object after a task is learn
        :param nn.Module model: the neural network in question
        :param torch.Dataloader dataloader: the training data for the task
    """
    def update(self, model, dataloader):
        mat_list = self.get_representation_matrix(model, dataloader)
        self.feature_list = self.update_GPM(mat_list, self.feature_list if hasattr(self, 'feature_list') else None)
        logger.debug('Feature list has %s elements' % str(sum([torch.tensor(x).numel() for x in self.feature_list])))
        self.feature_mat = self.get_feature_mat(model, self.feature_list)
        logger.debug('Feature mat has %s elements' % str(sum([torch.tensor(x).numel() for x in self.feature_mat])))

    """
        Applies GPM to the gradient, to prevent catastrophic forgetting
        :param torch.tensor grad: gradient of a layer of the neural network
        :param int k: to indicate to which layer the gradient belongs
    """
    def regularize(self, grad, k):
        return grad - torch.matmul(grad, self.feature_mat[k]).view(grad.size())


class KF_EWC:
    def __init__(self):
        self.Q = {}
        self.H = {}

    """
        Transform dictionary of tensors into 1D tensor
        :param dict param: dictionary of tensors
    """
    @staticmethod
    def to_vec(param):
        return torch.cat([p.view(-1) for n, p in param.items()])

    def update_KF(self, net_, dataloader):
        net_ = net_.to(device)
        criterion = nn.CrossEntropyLoss()
        for i, data in enumerate(dataloader, 0):
            net_.zero_grad()
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            outputs = net_(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            self.compute_HQ(net_, loss)
        self.old_net = {n: copy.deepcopy(p.detach()) for n, p in net_.named_parameters() if self.is_shared(n)}

    def compute_HQ(self, net_, loss):
        for layer in net_.layers_in_reverse_order:
            N = net_.state_dict()[layer + ".weight"].size(0)
            logger.debug('N for layer %s = %d' % (str(layer), N))
            hess = torch.autograd.functional.hessian(loss, net_.kf['pre-activation'][layer], create_graph=True)
            hess = torch.sum(hess[0], dim=0)
            H_old = 1.0 * H_new
            prev_layer = layer

    def compute_HQold(self, net_, loss):
        for layer in net_.layers_in_reverse_order:
            N = net_.state_dict()[layer + ".weight"].size(0)
            logger.debug('N for layer %s = %d' % (str(layer), N))
            hess = torch.autograd.functional.hessian(loss, net_.kf['pre-activation'][layer], create_graph=True)
            if H_old is None:
                hess = torch.autograd.functional.hessian(loss, net_.kf['pre-activation'][layer],
                                                         create_graph=True, allow_unused=True)
                logger.debug('Hess for layer %s: %s' % (str(layer), str(hess)))
                H_new = torch.zeros([N, N], device=device)
                for i in range(N):
                    H_new[i,i] = hess[i]
            else:
                dfdh = torch.autograd.grad(net_.kf['post-activation'][layer].sum(), net_.kf['pre-activation'][layer],
                                           create_graph=True, retain_graph=True, allow_unused=True)
                logger.debug('dfdh for layer %s: %s' % (str(layer), str(dfdh)))
                hess = torch.autograd.grad(dfdh.sum(), net_.kf['pre-activation'][layer],
                                           create_graph=True, retain_graph=True, allow_unused=True)
                logger.debug('Hess for layer %s: %s' % (str(layer), str(hess)))
                derror = torch.autograd.grad(loss, net_.kf['post-activation'][layer],
                                             create_graph=True, retain_graph=True, allow_unused=True)
                logger.debug('derror for layer %s: %s' % (str(layer), str(derror)))
                B, D = torch.zeros([N, N], device=device), torch.zeros([N, N], device=device)
                for i in range(N):
                    B[i,i] = dfdh[i]
                    D[i,i] = hess[i] * derror[i]
                H_new = torch.matmul(B, torch.matmul(torch.transpose(net_.state_dict()[layer]), 0, 1),
                                     torch.matmul(H_old, torch.matmul(net_.state_dict()[layer], B))) + D
            Q = torch.matmul(net_.kf['input'][layer].view(-1, 1), net_.kf['input'][layer].view(1, -1))
            try:
                self.H[layer] += H_new
                self.Q[layer] += Q
            except Exception as e:
                logger.warning('Exception when adding Q, H: %s' % str(e))
                self.H[layer] = H_new
                self.Q[layer] = Q
            H_old = 1.0 * H_new
            prev_layer = layer

    def regularize(self, current_net, alpha):
        y, x = torch.tensor([], device=device), torch.tensor([], device=device)
        for n, p in current_net.named_parameters():
            layer = n.split('.')[0]
            y = torch.cat((y, p.view(-1)))  # gather the (shared) parameters in a vector
            HWQ = torch.matmul(self.H[layer], torch.matmul(torch.transpose(p - self.old_net[n], 0, 1), self.Q[layer]))
            x = torch.cat((x, HWQ.view(-1)))
        return alpha / 2 * torch.dot(y, x)
