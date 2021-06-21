import torch
import torch.nn as nn
import copy
import numpy as np
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger = logging.getLogger('main')
logger.debug('Device in csqn_inv.py = %s' % str(device))


"""
    Continual Learning with Sampled-Quasi Newton
    :param str method: (optional) 'SR1' or 'BFGS' - default is 'SR1'
    :param list shared_layers: (optional) if provided, only layers in shared_layers are subjected to regularization
    :param int M: (optional) number of sampled points, determines rank of Hessian - default M=10
    :param float eps_ewc: (optional) to compute inverse FIM used for sampling - default is 1e-4
    :param str reduction: (optional) strategy to reduce Z - may be ['none', 'tree', 'split', 'once', 'auto', 'constant']
    :param float ev_ratio: (optional) explained variance ratio, only relevant if reduce in ['auto', 'split']
    :param int n_classes: (optional) required if multi-head classification
    :param float eps: (optional) to check the conditions for BFGS and SR1 when sampling - default is 1e-8
"""


class CSQN_Inv:
    def __init__(self, method='SR1', shared_layers=None, M=10, eps_ewc=1e-4, reduction='none', eps=1e-8,
                 ev_ratio=0.95, n_classes=None):
        self.is_shared = lambda x: shared_layers is None or x in shared_layers
        self.M = M
        self.red_strategy = reduction
        self.eps_ewc = eps_ewc
        self.eps = eps
        self.ev_ratio = ev_ratio
        self.method = method
        self.n_classes = n_classes
        logger.debug('CSQN with method = %s, M = %d, reduction = %s, eps_ewc = %s' % (method, M, reduction, str(eps_ewc)))
        self.task = 0

    """
        Returns the name of the CSQN object (given the method, number of components, etc.)
    """
    def get_name(self):
        name = 'CiSQN'
        if self.method == 'BFGS':
            name += '-B'
        else:
            name += '-S'
        if self.red_strategy != 'none':
            red_strat = {'auto': 'EV', 'split': 'SPL', 'tree': '2N', 'm': 'CT', 'once': '1N'}
            name += ' ' + red_strat[self.red_strategy]
        return name + ' (%d)' % self.M

    """
        Transforms a dictoinary or list of tensors to 1D tensor
        :param dict or list param: dict or list of tensors
    """
    @staticmethod
    def to_vec(param):
        if isinstance(param, dict):
            return torch.cat([p.view(-1) for n, p in param.items()])
        elif isinstance(param, list) or isinstance(param, tuple):
            return torch.cat([p.reshape(-1) for p in param])
        else:
            raise Exception("expected dictionary, list or tuple")

    """
        Computes the loss for a given number of batches
        :param nn.Module net_: neural network
        :param torch.Dataloader dataloader: the data as an iterator
        ;param int task: (optional) task id, only necessary if multi-head classification
        :param int steps: (optional) number of batches to consider for loss
    """
    def loss_fn(self, net_, dataloader, task=None, steps=5):
        criterion = nn.CrossEntropyLoss()
        loss, k, end = 0, 0, False
        for i in range(steps):
            try:
                inputs, labels = dataloader.next()
                if self.n_classes is not None:
                    labels = labels % self.n_classes
                outputs = net_(inputs.to(device), task) if task is not None else net_(inputs.to(device))
                loss += criterion(outputs, labels.to(device))
                k += 1
            except Exception as e:
                # exception triggered means dataloader has fewer than steps batches left
                logger.warning('Exception: iterator reached the end! - %s' % str(e))
                end = True  # reached the end of the iterator
                break
        return loss, end, k

    """ 
        Computes the diagonal of the FIM  
        :param nn.Module net_: the neural network
        :param torch.Dataloader dataset: the data
        :param int task: (optional) task id, required for multi-head classification
    """
    def compute_FIM(self, net_, dataset, task=None):
        iw = {n: torch.zeros_like(p, device=device) for n, p in net_.named_parameters() if self.is_shared(n)}
        criterion = nn.CrossEntropyLoss()
        net_ = net_.to(device)
        k = 0
        for i, data in enumerate(dataset, 0):
            net_.zero_grad()
            inputs, labels = data
            if self.n_classes is not None:
                labels = labels % self.n_classes
            outputs = net_(inputs.to(device), task) if task is not None else net_(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            for n, p in net_.named_parameters():
                if self.is_shared(n):
                    iw[n] = iw[n] + p.grad ** 2
            k += 1
        net_.cpu()
        return self.to_vec({n: p / k for n, p in iw.items()})


    """
        Computes X and A from S and Y when method = SR1
        :param torch.tensor S: an N times M tensor
        :param torch.tensor Y: an N times M tensor
        :param torch.tensor init: initial H
    """
    @staticmethod
    def compute_XA_from_SY_SR1(S, Y, init):
        M = Y.size(1)
        D, L, SBS = torch.zeros([M, M], device=device), torch.zeros([M, M], device=device), torch.zeros([M, M], device=device)
        X = torch.zeros_like(S, device=device)
        for i in range(M):
            D[i, i] = torch.dot(Y[:, i].to(device), S[:, i].to(device))
            for j in range(M):
                SBS[i, j] = torch.dot(Y[:, i].to(device), init * Y[:, j].to(device))
                if i > j:
                    L[i, j] = torch.dot(Y[:, i].to(device), S[:, j].to(device))
            X[:, i] = S[:, i].to(device) - init * Y[:, i].to(device)
        A = torch.inverse(D + L + torch.transpose(L, 0, 1) - SBS)
        return X, A

    """
        Computes Z from S and Y
        :param torch.tensor S: an N times M tensor
        :param torch.tensor Y: an N times M tensor
        :param torch.tensor ewc: diagonal of FIM as a 1D vector of size N
    """
    def compute_Z_from_SY(self, S, Y, ewc):
        X, A = self.compute_XA_from_SY_SR1(S, Y, ewc)
        logger.debug('Computing Z..')
        F, U = torch.eig(A.to(device), eigenvectors=True)  # computing eigenvalue decomposition of A
        logger.debug("Eigenvalues of A = %s", str(F))
        try:
            R = torch.cholesky(A, upper=False)  # Cholesky factorization if A is positive definite
        except:
            # exception means that A was not positive definite
            F[:, 0][F[:, 0] < 0] = 0  # Set negative eigenvalues to zero
            _, R = torch.qr(torch.transpose(U * torch.sqrt(F[:, 0]), 0, 1))  # compute QR factorization of (U*sqrt(F))'
            R = torch.transpose(R, 0, 1)
        Z = torch.matmul(X.cpu(), R.cpu())
        torch.cuda.empty_cache()
        return Z

    """
        
    """
    @staticmethod
    def IHVP_SR1(grad, init, Z):
        logger.debug('Share of init = %.4f; Share of Z = %.4f' % (torch.norm(init * grad),
                                                                  torch.norm(torch.matmul(torch.transpose(Z, 0, 1), grad))))
        return init * grad + torch.matmul(Z, torch.matmul(torch.transpose(Z, 0, 1), grad))


    """
        Computes the Hessian vector product with X, A
        :param torch.tensor vec: a 1D tensor of size N
        :param torch.tensor init: a 1D tensor of size N
        :param torch.tensor Z: a 2D tensor of size N times M
    """
    @staticmethod
    def IHVP_BFGS(vec, init, S, Y):
        q = 1.0 * vec
        alpha = torch.zeros(S.size(1), device=device)
        for i in range(S.size(1)-1, -1, -1):
            alpha[i] = torch.dot(S[:,i], q) / torch.dot(S[:,i], Y[:,i])
            q = q - alpha[i] * Y[:,i]
        r = init * q
        for i in range(S.size(1)):
            beta = torch.dot(Y[:,i], r) / torch.dot(Y[:,i], S[:,i])
            r = r + S[:,i] * (alpha[i] - beta)
        return r


    """
        Puts Z and init on the desired device
        :param bool cpu: (optional) if True, moves Z and init to cpu, else to device
    """
    def to_device(self, cpu=False):
        if not cpu:
            if hasattr(self, 'Z'):
                self.Z = self.Z.to(device)
            if hasattr(self, 'init'):
                self.init = [init.to(device) for init in self.init]
            if hasattr(self, 'x_old'):
                self.x_old = self.x_old.to(device)
            if hasattr(self, 'X'):
                self.X = self.X.to(device)
            if hasattr(self, 'A'):
                self.A = self.A.to(device)
            logger.debug("Moved to CUDA")
        else:
            if hasattr(self, 'Z'):
                self.Z = self.Z.cpu()
            if hasattr(self, 'init'):
                self.init = [init.cpu() for init in self.init]
            if hasattr(self, 'x_old'):
                self.x_old = self.x_old.cpu()
            if hasattr(self, 'X'):
                self.X = self.X.cpu()
            if hasattr(self, 'A'):
                self.A = self.A.cpu()
            logger.debug("Moved to CPU")

    """ 
        Check the conditions as described by Beharas et al. (s, y) pairs not satisfying the condition are removed
        :param torch.tensor S: the sampled S matrix
        :param torch.tensor Y: the corresponding Y matrix
        :param torch.tensor ewc: 1D-tensor representing the diagonal of B0
    """
    def check_conditions(self, S, Y):
        Snew, Ynew = None, None
        for i in range(self.M):
            if torch.dot(S[:,i], Y[:,i]) > self.eps * torch.dot(S[:,i], S[:,i]):
                try:
                    Snew, Ynew = torch.cat((Snew, S[:,i].view(1, -1)), 0), torch.cat((Ynew, Y[:,i].view(1, -1)), 0)
                except:
                    Snew, Ynew = S[:,i].view(1, -1), Y[:,i].view(1, -1)
        return torch.transpose(Snew, 0, 1).cpu(), torch.transpose(Ynew, 0, 1).cpu()


    """
        Samples the curvature pairs to construct S and Y
        :param nn.Module net_: the neural network
        :param torch.Dataloader dataset: the data
        :param torch.tensor std: 1D tensor of size N, standard deviation for sampling
        :param int task: (optional) task id, required when multi-head classification
    """
    def sample_curvature_pairs(self, net_, dataset, std, ewc, task=None):
        logger.debug('Sampling S, Y..')
        S, Y = None, None
        # sample M s vectors, combine them in matrix S
        for i in range(self.M):
            s = - torch.normal(mean=torch.zeros_like(std, device=device), std=std)
            try:
                S = torch.cat((S, s.view(1, -1)), 0)  # add vector to S
            except Exception as e:
                S = s.view(1, -1)  # S was None, initialize it with S
        S = torch.transpose(S, 0, 1).cpu()  # S is M*N, must be N*M
        net_ = net_.to(device)
        params = [p for n, p in net_.named_parameters() if self.is_shared(n)]  # parameter for which to compute gradient
        dataloader, Finished, K = iter(dataset), False, 0
        while not Finished:
            loss, Finished, k = self.loss_fn(net_, dataloader, task=task, steps=10)  # compute loss for 'steps' batches
            if k == 0:  # indicates that we have reached end of iterator dataloader
                break
            dfdpar = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True, allow_unused=True)
            dfdx = torch.tensor([], device=device)  # to combine gradient into a vector
            for x in dfdpar:
                dfdx = torch.cat((dfdx, x.view(-1)))
            for i in range(self.M):
                # compute Hessian-vector product with the ith S vector
                y = torch.autograd.grad(dfdx, params, grad_outputs=S[:, i].to(device),
                                        allow_unused=True, retain_graph=True)
                try:
                    Y[i, :] += self.to_vec(y)  # Add to Y
                except:
                    try:
                        Y = torch.cat((Y, self.to_vec(y).view(1, -1)), 0)  # Add column if Y has no ith column yet
                    except:
                        Y = self.to_vec(y).view(1, -1)  # Initialize Y
            del dfdx, dfdpar, loss, y
            torch.cuda.empty_cache()
            K += k  # k contains the number of batches in this step, K the total number
        Y = torch.transpose(Y / K, 0, 1).cpu()
        S, Y = self.check_conditions(S.to(device), Y.to(device))  # we check if S, Y satisfy the conditions
        logger.debug("Size of S = %s" % str(Y.size()))
        logger.debug("Size of Y = %s" % str(Y.size()))
        return S.to(device), Y.to(device)

    """
        Computes the regularization loss for CSQN
        :param nn.Module current_net: the neural network subject to regularization
        :param float alpha: the weight of the regularization
    """
    def regularize(self, grad):
        K = self.S.size(1) // self.M if hasattr(self, 'S') else self.Z.size(1) // self.M
        for i in range(K):
            logger.debug('Regularization for task %d..' % i)
            grad_norm = torch.norm(grad)
            if self.method == 'BFGS':
                grad = self.IHVP_BFGS(grad, self.init[i],
                                       self.S[:,i*self.M:(i+1)*self.M], self.Y[:,i*self.M:(i+1)*self.M])
            else:
                grad = self.IHVP_SR1(grad, self.init[i], self.Z[:,i*self.M:(i+1)*self.M])
            logger.debug('...grad before vs. grad now = %.4f vs. %.4f' % (grad_norm, torch.norm(grad)))
            grad = grad / torch.norm(grad) * grad_norm
        return grad


    """
        Updates the CSQN object: computes S, Y and Z and combines the new and the old Z
        :param nn.Module net_: the neural network to compute S and Y and use in regularization
        :param torch.Dataloader dataset: the dataset
        :param int task: (optional) task id, required if multi-head classification
    """
    def update(self, net_, dataset, task=None):
        self.to_device(cpu=True)
        ewc = self.compute_FIM(net_, dataset, task)  # compute the FIM
        #logger.debug('Norm of EWC = %.4f' % torch.norm(ewc))
        inv_ewc = 1 / (ewc + self.eps_ewc * ewc.max())
        std = torch.sqrt(inv_ewc)  # std used for sampling S
        #std = torch.ones(sum([p.numel() for n, p in net_.state_dict().items() if self.is_shared(n)]))
        S, Y = self.sample_curvature_pairs(net_, dataset, ewc=None, std=std, task=task)
        #gamma = sum([torch.dot(S[:,i], Y[:,i]) / torch.dot(Y[:,i], Y[:,i]) for i in range(self.M)]) / self.M
        #init = gamma * torch.ones_like(ewc)
        init = inv_ewc
        if hasattr(self, 'init'):
            self.init.append(init)
        else:
            self.init = [init]
        if self.method == 'BFGS':
            if hasattr(self, 'S') and hasattr(self, 'Y'):
                self.S, self.Y = torch.cat((self.S, S), dim=1), torch.cat((self.Y, Y), dim=1)
            else:
                self.S, self.Y = S, Y
        else:
            Z = self.compute_Z_from_SY(S, Y, init)
            if hasattr(self, 'Z'):
                self.Z = torch.cat((self.Z, Z), dim=1)
            else:
                self.Z = Z
        logger.info("Finished training!")
        self.to_device()
        self.task += 1

