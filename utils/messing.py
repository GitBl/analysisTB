import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def messing_up(module):
    """
    Shifts the weigth by 0.1.
    """
    if type(module) == nn.Conv2d:
        if not hasattr(module, 'first'):  # Do not mess up the first Conv2D, else it's unfair
            torch.nn.init.kaiming_normal_(module.weight)
            module.weight = nn.Parameter(module.weight + 0.1)


def zero_mess(module):
    """
    Initiate the weight not tagged by the first attribute
    """
    if type(module) == nn.Conv2d:
        if not hasattr(module, 'first'):  # Do not mess up the first Conv2D, else it's unfair
            torch.nn.init.zeros_(module.weight)


def var_modifier(tensor, val):
    """
    Modifies the variance of the given Tensor by the input constant value.
    """
    with torch.no_grad():
        return tensor.mul_(np.sqrt(val))


def non_res_reslike_initialization(scaling):
    """
    Initialize a non-resnet convolutional network according to a resnet type initialization 
    """
    def init_pattern(module):
        if(type(module)) == nn.Conv2d:
            # Do not mess up the first Conv2D, else it's unfair
            if not hasattr(module, 'first'):
                init_matrix = torch.zeros(module.weight.size())
                k_size = module.weight.size()[2]

                init_weight = torch.zeros(k_size, k_size)

                scaling_value = 1
                if(scaling):
                    scaling_value = module.in_channels

                init_weight[(k_size-1) // 2][(k_size-1) // 2] = (1. +
                                                                 np.random.uniform(-10**-9, 10**-9))/scaling_value

                for in_channel in range(module.weight.size()[0]):
                    for out_channel in range(module.weight.size()[1]):
                        init_matrix[in_channel][out_channel] = init_weight
                module.weight = nn.Parameter(init_matrix)
    return init_pattern

# def variance_modifier(value):
#    if type(module) == nn.Conv2d:
#        if hasattr(module, 'tagged'):
#            var_modifier(module.weight)
#    return variance_modifier


def mess_generator(value):
    """
    Returns a messing function offsetting every non "first" layer by value.

    Is particularly helpful when using net.apply(mess_generator(value))0
    """
    def mess(module):
        if type(module) == nn.Conv2d:
            # Do not mess up the first Conv2D, else it's unfair
            if not hasattr(module, 'first'):
                torch.nn.init.kaiming_normal_(module.weight)
                module.weight = nn.Parameter(module.weight + value)
    return mess


def zero_init(net, verbose=False, iteration_max=float('inf')):
    iteration = 0
    for module in net.modules():
        tagged = False
        if(hasattr(module, "in_channels") and hasattr(module, "out_channels")):
            iteration += 1
            if(module.in_channels == module.out_channels and iteration < iteration_max):
                if(module.kernel_size[0] != 1 and module.stride == (1, 1)):
                    if(verbose):
                        print(module.kernel_size)
                    tagged = True
        if(not tagged):
            module.first = True

    net.apply(zero_mess)

    if(verbose):
        for name, weight in net.named_parameters():
            if(torch.sum(weight) == 0 and "conv" in name and not 'bias' in name):
                print(name)
