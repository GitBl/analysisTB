import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


def plot_specific_weight(curr_list, value):
    plot_list = []
    for element in curr_list:
        plot_list.append(element[value//3][value % 3])

    plt.figure()
    plt.title("Weigth n°{}".format(value))
    plt.hist(plot_list, bins=50)
    plt.show()


def weight_reporter(net):
    param_list = []
    for name, param in net.named_parameters():
        if (".conv.weight" in name):
            in_layer, out_layer, kernel_size, kernel_size = param.size()

            for i in range(in_layer):
                for j in range(out_layer):
                    param_list.append(param[i][j].cpu().detach())
    for i in range(9):
        plot_specific_weight(param_list, i)


def comparaison_plot(out, label_list=None, name_list=None, NB_EPOCH=1, plot_text=""):
    """
    Plot the result of the comparaison of two networks
    """
    plt.style.use("ggplot")

    for i in range(len(out)):
        plt.figure()
        if(label_list):

            epoch_list = np.linspace(0, NB_EPOCH, len(out[i][0]))

            plt.plot(epoch_list, out[i][0], label=name_list[0])
            plt.plot(epoch_list, out[i][1], label=name_list[1])
        plt.title(label_list[i]+" " + plot_text)
        plt.xlabel("Epochs")
        plt.ylabel(label_list[i])
        plt.legend()
        plt.show()


def bias_inspector(net, return_name=False):
    if(return_name):
        name_list = []
    parameter_list = []
    for name, param in net.named_parameters():
        if('bias' in name and not 'fc' in name and not 'lin' in name):
            if(return_name):
                name_list.append(name)
            parameter_list.append(param.cpu().detach().numpy())
    return_list = parameter_list
    if(return_name):
        return_list = [return_list]
        return_list.append(name_list)
    return return_list


def LRVarshow(var_range, lr_range, out_val_list, epoch_number=None, model_name=None):
    for out_val in out_val_list:
        fig = plt.figure()
        ax = Axes3D(fig)

        x_range = len(var_range)
        y_range = len(lr_range)

        X = np.zeros((x_range, y_range))
        Y = np.zeros((x_range, y_range))

        for i in range(x_range):
            for j in range(y_range):
                X[i][j] = i
                Y[i][j] = j

        ax.set_xlabel('Variance Coefficient')
        ax.set_ylabel('LR')
        ax.set_zlabel('Test Score')
        ax.plot_surface(X, Y, out_val, cmap='seismic', alpha=0.7)
        plt.yticks(range(y_range), lr_range)
        plt.xticks(range(x_range), var_range)
        title = ""
        if(epoch_number):
            title = title + "Epoch(s): " + str(epoch_number) + " "
        if(model_name):
            title = title + str(model_name)
        if(title != ""):
            plt.title(title)
        plt.show()


def CKAmatshow(matrix):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    plt.show()


def dictionnary_show(dictionnary, key_list, LR_list, plotting=False, epoch_analysis=False, return_param=False):
    nb_epochs = dictionnary["nb_epochs"]
    ppe = dictionnary["points_per_epochs"]

    if return_param:
        return_mat = np.zeros((len(LR_list), len(key_list)))
    for LR in LR_list:
        for plot in key_list:
            if(plotting):
                plt.figure()
            for element in dictionnary:
                param = element.split("/")
                if(len(param) != 1):
                    plt.style.use("ggplot")
                    if(param[1] == str(LR) and param[-1] == plot):

                        if(plotting):
                            plt.title(param[1])
                            plt.ylabel(param[-1])

                            epoch_list = np.arange(
                                nb_epochs, step=1/(ppe))
                            plt.xlabel("Epochs")

                        if(plot == "loss"):
                            interesting_value = min(dictionnary[element])

                            if(epoch_analysis):
                                for i in range(nb_epochs):
                                    print("\t \t \t \t Epochmax :{}, value: {:.2f}".format(
                                        i+1, min(dictionnary[element][:int((i+1)*ppe)])))

                        if(plot == "score"):
                            interesting_value = max(dictionnary[element])

                            if(epoch_analysis):
                                for i in range(nb_epochs):
                                    print("\t \t \t \t Epochmax :{}, value: {:.2f}".format(
                                        i+1, max(dictionnary[element][:int((i+1)*ppe)])))

                        if(plotting):
                            plt.plot(
                                epoch_list, dictionnary[element], label=param[0] + " ,{:.2f}".format(interesting_value))
                        else:
                            print("{} - : {:.2f}".format(element, interesting_value))
                            print(
                                "--------------------------------------------------------")
                            print("")
            if(plotting):
                plt.legend()
            if(plotting):
                plt.show()
