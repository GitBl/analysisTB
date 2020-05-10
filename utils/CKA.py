import torch.nn as nn
import math
from torch import cuda
import numpy as np
import torch
# According to paper - Include parts from https://github.com/google-research/google-research/tree/master/representation_similarity


def kernalize(X, cbf=True, sigma=1):
    # if (type(X) == torch.Tensor):
    #        X = X.detach().numpy()

    if(cbf):

        proj_mat = X.dot(X.T)
        first_part = np.diag(proj_mat) - proj_mat
        final_mat = first_part + first_part.T
        final_mat *= -1/(2*(sigma ** 2))
        print("Final mat : {}".format(final_mat))
        return np.exp(final_mat)
    else:
        return X.dot(X.T)


def gram_centering(gram):
    if(type(gram) == torch.Tensor):
        means = torch.mean(gram, 0, dtype=torch.float32)
        means -= torch.mean(means) / 2
    else:
        means = np.mean(gram, 0, dtype=np.float64)
        means -= np.mean(means) / 2
    gram -= means[:, None]
    gram -= means[None, :]
    return gram


def CKA(X, Y, cbf, sigma=1, verbose=False):
    K = gram_centering(googlegram_rbf(X))  # ,cbf,sigma))
    L = gram_centering(googlegram_rbf(Y))  # ,cbf,sigma))

    numerator = HSIC(K, L)

    first_h = HSIC(K, K)
    second_h = HSIC(L, L)

    if verbose:
        print("{}, {}, {}".format(numerator, first_h, second_h))
        if(first_h == 0):
            print("K : {}".format(K))
            print("pure K : {}".format(kernalize(X, cbf, sigma)))
            print(X)
        if(second_h == 0):
            print("L : {}".format(L))
            print(Y)
    if(first_h == 0 or second_h == 0 or numerator == 0):
        return 0
    if(type(first_h) == torch.Tensor):
        return numerator/(torch.sqrt(first_h * second_h))
        del X
        del Y
    else:
        return numerator/(np.sqrt(first_h * second_h))


def HSIC(K, L):
    n = K.shape[0]

    if(type(K) == torch.Tensor):
        H = torch.from_numpy(np.eye(n) - np.ones((n, n))/n).float()
        final_mat = (K.mm(H)).mm(L.mm(H))

        return (torch.trace(final_mat))  # /(n-1)**2)
    else:
        H = np.eye(n) - np.ones((n, n))/n
        final_mat = (K.dot(H)).dot(L.dot(H))

        return (np.trace(final_mat))  # /(n-1)**2)


def CKA_net_computation(network, dataset, cbf=True, sigma=1, verbose=False, fast_computation=False, iteration_limit=10):
    """
    Returns the CKA matrix for the input networks, thanks to the Google algorithms.

    Excpect a matrix of size (nn.Conv layers size*nn.Conv layers size)

    cbf: Whether or not to use RBF kernel
    sigma: which sigma to use for the RBF kernel
    fast_computation: take only "iteration_limit" batchs for early results
    """
    if (next(network.parameters()).is_cuda):  # CUDA Trick
        network = network.cpu()

    linking_list = []

    for module in network.modules():
        if type(module) == nn.Conv2d:
            linking_list.append(module)

    hook_value = [-1]*len(linking_list)

    n = len(linking_list)

    def registering_hook(self, in_val, out_val):
        to_store = in_val[0]
        to_store = to_store.view(
            to_store.shape[0], np.product(to_store.shape[1:]))
        hook_value[linking_list.index(self)] = to_store

    for module in network.modules():
        if type(module) == nn.Conv2d:
            module.register_forward_hook(registering_hook)

    return_matrix = torch.zeros((n, n))

    # Dataset pass
    if(fast_computation):
        iteration = 0
        for batch, _ in dataset:
            if(iteration < iteration_limit):
                iteration += 1
                network(batch)
                for i in range(n):
                    for j in range(i+1):
                        # print(hook_value[i])
                        temp = CKA(hook_value[i], hook_value[j],
                                   cbf, sigma, verbose)/iteration_limit
                        return_matrix[i][j] += temp
                        del temp
                print("Done: {:.2f}".format(
                    100*(iteration/(iteration_limit))), end='\r')
        for i in range(n):
            for j in range(i, n):
                return_matrix[i, j] = return_matrix[j, i]
    else:
        iteration = 0
        for batch, _ in dataset:
            iteration += 1
            network(batch)
            for i in range(n):
                for j in range(i+1):
                    # print(hook_value[i])
                    temp = CKA(hook_value[i], hook_value[j],
                               cbf, sigma, verbose)/len(dataset)
                    return_matrix[i][j] += temp
                    del temp
            print("Done: {:.2f}".format(
                100*(iteration/(len(dataset)+1))), end='\r')
        for i in range(n):
            for j in range(i, n):
                return_matrix[i, j] = return_matrix[j, i]

    return return_matrix


def googlegram_rbf(x, threshold=1.0):
    if (type(x) == torch.Tensor):
        dot_products = x.mm(torch.transpose(x, 0, 1))
        sq_norms = torch.diag(dot_products)
        sq_distances = -2 * dot_products + \
            sq_norms[:, None] + sq_norms[None, :]
        sq_median_distance = torch.median(sq_distances)
        return torch.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))
    else:
        dot_products = x.dot(x.T)
        sq_norms = np.diag(dot_products)
        sq_distances = -2 * dot_products + \
            sq_norms[:, None] + sq_norms[None, :]
        sq_median_distance = np.median(sq_distances)
        return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))
