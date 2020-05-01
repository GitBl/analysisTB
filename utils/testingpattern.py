import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt
import utils.plotting_tools as plt_tools
import utils.messing as messing
import utils.engine as engine
import copy
import torchvision


def testing_pattern(network, train_loader, test_loader, LR_list, points_per_epochs=10, nb_epochs=5, naive=True):

    cuda_state = torch.cuda.is_available()

    initialization_list = ["Smart", "Test"]

    if(naive):
        initialization_list.append("Naïve")

    print(initialization_list)
    return_dict = {}

    return_dict["points_per_epochs"] = points_per_epochs
    return_dict["nb_epochs"] = nb_epochs

    for initialization_pattern in initialization_list:
        for LR in LR_list:

            current_network = copy.deepcopy(network)

            if(cuda_state):
                current_network = current_network.cuda()

            if(initialization_pattern == "Naïve"):
                for module in current_network.modules():
                    if(type(module) == nn.Conv2d):
                        torch.nn.init.zeros_(module.weight)
            elif(initialization_pattern == "Smart"):
                for module in current_network.modules():
                    if(type(module) == nn.Conv2d):
                        if(module.in_channels == module.out_channels):
                            if(module.kernel_size[0] != 1):
                                torch.nn.init.uniform_(
                                    module.weight, -10**-5, 10**-5)

            curr_loss_list = []
            curr_score_list = []

            current_optim = torch.optim.SGD(current_network.parameters(), LR)

            CRITERION = nn.CrossEntropyLoss()

            for epoch_num in range(nb_epochs):
                # Training pass
                for i, data in enumerate(train_loader):

                    image = data[0].type(torch.FloatTensor).cuda()
                    label = data[1].type(torch.LongTensor).cuda()

                    pred_label = current_network(image)
                    if(type(network) == torchvision.models.GoogLeNet):
                        pred_label = pred_label.logits
                    loss = CRITERION(pred_label, label)

                    current_optim.zero_grad()
                    loss.backward()
                    current_optim.step()

                    if(int((i - 1) * points_per_epochs / len(train_loader)) != int(i * points_per_epochs / len(train_loader)) or i == len(train_loader) - 1):
                        curr_loss_list.append(loss.data.item())
                        curr_score_list.append(engine.get_score(
                            current_network, test_loader))

                        print("Technique: {}, LR: {}, Done: {}%, epoch:{}/{}, score = {:.2f}, loss = {:.2f} ".format(initialization_pattern,
                                                                                                                     LR, int(100.*(i - 1) / len(train_loader)), epoch_num+1, nb_epochs, curr_score_list[-1], curr_loss_list[-1]), end="\r")

            curr_key = initialization_pattern + "/"+str(LR)

            return_dict[curr_key + "/loss"] = curr_loss_list
            return_dict[curr_key + "/score"] = curr_score_list
            print("")
            del current_network

    return return_dict
