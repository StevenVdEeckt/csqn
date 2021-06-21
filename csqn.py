import torch
import torch.nn as nn
import copy
import numpy as np
import logging

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logger = logging.getLogger('main')
logger.debug('Device in csqn.py = %s' % str(device))


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


class CSQN:
    def __init__(self, method='SR1', shared_layers=None, M=10, reduction='none', n_classes=None, eps_ewc=1e-4,
                 eps=1e-8, ev_ratio=0.95):
        self.is_shared = lambda x: shared_layers is None or x in shared_layers
        self.M = M
        self.red_strategy = reduction
        self.eps_ewc = eps_ewc
        self.eps = eps
        self.ev_ratio = ev_ratio
        self.method = method
        self.n_classes = n_classes
        logger.debug('CSQN with method = %s, M = %d, reduction = %s, eps_ewc = %s'
                     % (method, M, reduction, str(eps_ewc)))
        self.task = 0

    """
        Returns the name of the CSQN object (given the method, number of components, etc.)
    """
    def get_name(self):
        name = 'CSQN'
        if self.method == 'BFGS':
            name += '-B'
        else:
            name += '-S'
        if self.red_strategy != 'none':
            red_strat = {'auto': 'EV', 'split': 'SPL', 'tree': '2N', 'm': 'CT', 'once': '1N'}
            name += ' ' + red_strat[self.red_strategy]
        return name + ' (%d)' % self.M

    """
        Transforms a dictionary or list of tensors into 1D tensor
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
        Computes X and A from S and Y when method = BFGS
        :param torch.tensor S: an N times M tensor
        :param torch.tensor Y: an N times M tensor
        :param torch.tensor ewc: diagonal of FIM as a 1D vector of size N
    """
    @staticmethod
    def compute_XA_from_SY_BFGS(S, Y, ewc):
        N, M = Y.size()
        L, D, A = torch.zeros([M, M]).to(device), torch.zeros([M, M]).to(device), torch.zeros([2 * M, 2 * M]).to(device)
        SS, SY = torch.zeros([M, M]).to(device), torch.zeros([N, 2 * M]).to(device)
        for i in range(M):
            D[i, i] = torch.dot(S[:, i].to(device), Y[:, i].to(device))
            SY[:, i], SY[:, i + M] = ewc * S[:, i].to(device), Y[:, i].to(device)
            for j in range(M):
                SS[i, j] = torch.dot(S[:, i].to(device), ewc * S[:, j].to(device))
                if i > j:
                    L[i, j] = torch.dot(S[:, i].to(device), Y[:, j].to(device))
        A[:M, :M], A[:M, M:2 * M], A[M:2 * M, :M], A[M:2 * M, M:2 * M] = SS, L, torch.transpose(L, 0, 1), -D
        return SY, torch.inverse(-A)

    """
        Computes X and A from S and Y when method = SR1
        :param torch.tensor S: an N times M tensor
        :param torch.tensor Y: an N times M tensor
        :param torch.tensor ewc: diagonal of FIM as a 1D vector of size N
    """
    @staticmethod
    def compute_XA_from_SY_SR1(S, Y, ewc):
        M = Y.size(1)
        D, L, SBS = torch.zeros([M, M], device=device), torch.zeros([M, M], device=device), \
                    torch.zeros([M, M], device=device)
        X = torch.zeros_like(Y, device=device)
        for i in range(M):
            D[i, i] = torch.dot(S[:, i].to(device), Y[:, i].to(device))
            for j in range(M):
                SBS[i, j] = torch.dot(S[:, i].to(device), ewc * S[:, j].to(device))
                if i > j:
                    L[i, j] = torch.dot(S[:, i].to(device), Y[:, j].to(device))
            X[:, i] = Y[:, i].to(device) - ewc * S[:, i].to(device)
        A = torch.inverse(D + L + torch.transpose(L, 0, 1) - SBS)
        return X, A

    """
        Computes Z from S and Y
        :param torch.tensor S: an N times M tensor
        :param torch.tensor Y: an N times M tensor
        :param torch.tensor ewc: diagonal of FIM as a 1D vector of size N
        :param bool return_xa: (optional) if True, returns X, A instead of Z
    """
    def compute_Z_from_SY(self, S, Y, ewc, return_xa=False):
        if self.method == 'SR1':
            X, A = self.compute_XA_from_SY_SR1(S, Y, ewc)
        else:
            X, A = self.compute_XA_from_SY_BFGS(S, Y, ewc)
        if return_xa:
            return X, A
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
        Applies Singular Value Decomposition to Z to reduce the size of Z
        :param torch.tensor Z: an N times (2)M tensor
        :param bool ev: (optional) True if explained variance ratio must be considered, otherwise reduces to  size M
    """
    def apply_SVD(self, Z, ev=True):
        if Z.size(0) == 0:
            return Z
        U, E, V = torch.linalg.svd(Z.to(device), full_matrices=False)  # compute SVD of Z
        expl_var = torch.cumsum(E ** 2 / (E ** 2).sum(), dim=0)
        logger.debug("Explained variance ratio: %s" % str(expl_var))
        if not ev:
            K = self.M
        else:
            K = (expl_var < self.ev_ratio).sum().item() + 1
        logger.debug("Z will be reduced from %d to %d" % (Z.size(1), K))
        Z = np.sqrt(1.0 * Z.size(1) / K) * torch.matmul(U[:, :K] * E[:K], V[:K, :K]).cpu()
        return Z

    """
        Combine the previous Z and the Z from the current task
        :param torch.tensor Z: an N times (2)M tensor
        :param torch.tensor oldZ: an N times (2)M tensor
    """
    def combine_new_old_Z(self, Z, oldZ):
        if self.red_strategy == 'none':
            # do not reduce: combine old and new Z
            return torch.cat((oldZ, Z), dim=1)
        elif self.red_strategy == 'split':
            # reduce old Z and new Z separately, then combine
            logger.debug("For new Z...")
            Z = self.apply_SVD(Z, ev=True)
            logger.debug("For old Z..")
            oldZ = self.apply_SVD(oldZ, ev=True)
            return torch.cat((oldZ, Z), dim=1)
        elif self.red_strategy == 'once':
            # only reduce new Z, then combine with old Z
            Z = self.apply_SVD(Z, ev=True)
            return torch.cat((oldZ, Z), dim=1)
        elif self.red_strategy == 'auto':
            # combine and reduce new and old Z considering explained variance ratio
            return self.apply_SVD(torch.cat((oldZ, Z), dim=1), ev=True)
        elif self.red_strategy == 'm':
            # combine and reduce new and old Z such that its size remains constant (M)
            return self.apply_SVD(torch.cat((oldZ, Z), dim=1), ev=False)
        elif self.red_strategy == 'tree':
            # tree-structured reduction
            # no reduction if task is odd numbered
            if (self.task + 1) % 2 != 0:
                logger.debug("Odd task: no reduction")
                return torch.cat((oldZ, Z), dim=1)
            # reduction if even task
            logger.debug("Even task: reduction")
            j = self.task + 1  # assuming task 0 is the first task
            # keep reducing as long as j is a multiple of 2: when j is a power of 2, Z will have size M at the end
            while j % 2 == 0:
                logger.debug("Reduce Z..")
                Z = self.apply_SVD(torch.cat((oldZ[:, -self.M:], Z), dim=1), ev=False)
                oldZ = oldZ[:, :-self.M]
                j = j // 2
            return torch.cat((oldZ, Z), dim=1)

    """
        Combine the previous X, A and the X, A from the current task
        :param torch.tensor Xold: an N times (2)M tensor
        :param torch.tensor Aold: an (2)M times (2)M tensor
        :param torch.tensor X: an N times (2)M tensor
        :param torch.tensor A: an (2)M times (2)M tensor
        :param int M_: equal to M or 2M (depending on method)
    """
    @staticmethod
    def combine_new_old_XA(Xold, Aold, X, A):
        Anew = torch.zeros([Aold.size(0) + A.size(0), Aold.size(1) + A.size(1)])
        Anew[:Aold.size(0), :Aold.size(1)] = Aold
        Anew[Aold.size(0):, Aold.size(1):] = A
        return torch.cat((Xold, X), dim=1), Anew

    """
        Computes Z, combining the new and old Z. If method == 'BFGS' and reduction == 'none', then X, A are computed
        :param torch.tensor S: an N times M tensor
        :param torch.tensor Y: an N times M tensor
        :param torch.tensor ewc: diagonal of FIM as a 1D vector of size N
    """
    def compute_Z(self, S, Y, ewc):
        # if method is BFGS and we do not reduce, then we do not compute Z, but X, A (to not lose any information)
        if self.method == 'BFGS' and self.red_strategy == 'none':
            X, A = self.compute_Z_from_SY(S, Y, ewc, return_xa=True)
            if hasattr(self, 'X') and hasattr(self, 'A'):
                self.X, self.A = self.combine_new_old_XA(self.X.to(device), self.A.to(device), X, A)
            else:
                self.X, self.A = X, A
            logger.debug("Size of X: %s", str(self.X.size()))
            logger.debug("Size of A: %s", str(self.A.size()))
        # in all other cases, we compute Z
        else:
            Z = self.compute_Z_from_SY(S, Y, ewc)
            logger.debug("Z for current task = %s", str(Z.size()))
            if hasattr(self, 'Z') and self.Z is not None:
                oldZ = self.Z
                logger.debug("Z from previous tasks = %s", str(oldZ.size()))
            else:
                logger.debug("No Z from old tasks found...")
                oldZ = torch.tensor([])
            self.Z = self.combine_new_old_Z(Z, oldZ)
            logger.debug("Size of Z: %s", str(self.Z.size()))

    """
        Computes the Hessian vector product with Z
        :param torch.tensor vec: a 1D tensor of size N
        :param torch.tensor init: a 1D tensor of size N
        :param torch.tensor Z: a 2D tensor of size N times M
    """
    @staticmethod
    def HVP_Z(vec, init, Z):
        return init * vec + torch.matmul(Z, torch.matmul(torch.transpose(Z, 0, 1), vec))

    """
        Computes the Hessian vector product with X, A
        :param torch.tensor vec: a 1D tensor of size N
        :param torch.tensor init: a 1D tensor of size N
        :param torch.tensor Z: a 2D tensor of size N times M
    """
    @staticmethod
    def HVP_XA(vec, init, X, A):
        return init * vec + torch.matmul(X, torch.matmul(A, torch.matmul(torch.transpose(X, 0, 1), vec)))

    """
        Puts Z and init on the desired device
        :param bool cpu: (optional) if True, moves Z and init to cpu, else to device
    """
    def to_device(self, cpu=False):
        if not cpu:
            if hasattr(self, 'Z'):
                self.Z = self.Z.to(device)
            if hasattr(self, 'init'):
                self.init = self.init.to(device)
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
                self.init = self.init.cpu()
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
    def check_conditions(self, S, Y, ewc):
        Snew, Ynew = None, None
        if self.method == 'SR1':
            for i in range(self.M):
                X, A = self.compute_XA_from_SY_SR1(S[:, :i], Y[:, :i], ewc)
                Bs = self.HVP_XA(S[:, i], ewc, X.to(device), A.to(device))
                if abs(torch.dot(S[:, i], Y[:, i] - Bs)) > self.eps * torch.dot(S[:, i], S[:, i]):
                    try:
                        Snew, Ynew = torch.cat((Snew, S[:, i].view(1, -1)), 0), torch.cat((Ynew, Y[:, i].view(1, -1)), 0)
                    except:
                        Snew, Ynew = S[:, i].view(1, -1), Y[:, i].view(1, -1)
        else:
            for i in range(self.M):
                if torch.dot(S[:, i], Y[:, i]) > self.eps * torch.dot(S[:, i], S[:, i]):
                    try:
                        Snew, Ynew = torch.cat((Snew, S[:, i].view(1, -1)), 0), torch.cat((Ynew, Y[:, i].view(1, -1)), 0)
                    except:
                        Snew, Ynew = S[:, i].view(1, -1), Y[:, i].view(1, -1)
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
            except:
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
        S, Y = self.check_conditions(S.to(device), Y.to(device), ewc)  # we check if S, Y satisfy the conditions
        logger.debug("Size of S = %s" % str(Y.size()))
        logger.debug("Size of Y = %s" % str(Y.size()))
        return S.to(device), Y.to(device)

    """
        Computes the regularization loss for CSQN
        :param nn.Module current_net: the neural network subject to regularization
        :param float alpha: the weight of the regularization
    """
    def regularize(self, current_net, alpha):
        y = torch.tensor([], device=device)
        for n, p in current_net.named_parameters():
            if self.is_shared(n):
                y = torch.cat((y, p.view(-1)))  # gather the (shared) parameters in a vector
        if hasattr(self, 'Z'):
            return alpha * torch.dot(y - self.x_old, self.HVP_Z(y - self.x_old, self.init, self.Z))
        else:
            return alpha * torch.dot(y - self.x_old, self.HVP_XA(y - self.x_old, self.init, self.X, self.A))

    """
        Updates the CSQN object: computes S, Y and Z and combines the new and the old Z
        :param nn.Module net_: the neural network to compute S and Y and use in regularization
        :param torch.Dataloader dataset: the dataset
        :param int task: (optional) task id, required if multi-head classification
    """
    def update(self, net_, dataset, task=None):
        self.to_device(cpu=True)
        ewc = self.compute_FIM(net_, dataset, task)  # compute the FIM
        logger.debug('Norm of EWC = %.4f' % torch.norm(ewc))
        std = torch.sqrt(1 / (ewc + self.eps_ewc * ewc.max()))  # std used for sampling S
        S, Y = self.sample_curvature_pairs(net_, dataset, ewc=ewc, std=std, task=task)
        self.compute_Z(S, Y, ewc)  # computes Z for S-LSR1; X, A for S-LBFGS
        logger.info("Finished training!")
        self.init = ewc.cpu() + self.init if hasattr(self, 'init') else ewc  # initial Hessian estimate, i.e. B0
        logger.debug("Norm of init = %.4f" % (torch.norm(self.init)))
        self.x_old = self.to_vec({n: copy.deepcopy(p.detach()) for n, p in net_.named_parameters() if self.is_shared(n)})
        del S, Y, ewc
        self.to_device()
        torch.cuda.empty_cache()
        self.task += 1
