"""
This is designed to embed the major computation of algorithms.

"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import matplotlib.pyplot as plt
import utils.plotting_tools as plt_tools
import utils.messing as messing
import torchvision
from matplotlib import cm


def compute_grad_list_diff_score(list_1, list_2):
    """
    Compute the L1 value of the difference between two lists of parameter vectors (list of Tensors)
    """
    if(len(list_1) != len(list_2)):
        raise ValueError("Entry list should be of the same size")

    diff_list = [0] * len(list_1)
    for i in range(len(list_1)):
        diff_list[i] = list_1[i] - list_2[i]

    diff_list = [elem.flatten() for elem in diff_list]

    score = 0
    for i in range(len(diff_list)):
        score += np.sum(np.abs(diff_list[i])) / (diff_list[i].shape[0])
    return score


def get_grad_norm(net):
    """
    Return the norm of the gradient of a specific network
    """
    grad_list = []
    for elem in net.parameters():
        if(hasattr(elem, 'grad')):
            grad_list.append(elem.grad)
    grad_list = [torch.flatten(elem) for elem in grad_list]

    norm = 0
    total_length = 0

    for elem in grad_list:
        norm += torch.mean(torch.abs(elem)).item() * elem.shape[0]
        total_length += elem.shape[0]

    norm = norm / total_length

    return norm


def compute_loss(net, train_loader, criterion=nn.CrossEntropyLoss()):
    """
    Return the value of the loss for the network over the given loader
    """

    loss_value = 0
    for image, label in train_loader:
        pred_label = net(image.cuda())
        loss = criterion(pred_label, label.cuda())
        loss_value += loss.data.item() / len(train_loader)
        del pred_label
        torch.cuda.empty_cache()
    return loss_value


def get_score(net, dataset):
    """
    Return the score of a given network over a given dataset
    """
    test_score = 0.
    for data, label in dataset:
        pred = net(data.cuda())
        if(type(net) == torchvision.models.GoogLeNet):
            pred = pred.logits
        test_score = test_score + \
            torch.sum((torch.max(pred.cpu(), 1)[1] == label))
    return float(test_score) / (float(len(dataset)) * dataset.batch_size)


def weight_projection(weight_1, weight_2):
    """
    For two vector of weights of the same size, compute their naïve scalar product and returns it
    """
    return [
        torch.dot(
            weight_1[i].flatten(),
            weight_2[i].flatten()).item() for i in range(
            weight_1.shape[0])]


def complex_norm(vector):
    """
    Computes the norm of a weight vector using the scalar product defined earlier
    """
    return np.mean(weight_projection(vector, vector))


def GSplot(net, train_loader, learning_rate=None, GSrange=2, GSratio=1):
    """
    Stand for Grid Search Plot.

    When called, take two gradient step of the given train_loader.

    /!\ -> Not final and unstable
    """
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(net.parameters(), learning_rate)
    iteration = 0
    gradient_list = []

    for image, label in train_loader:
        if (iteration < 2):
            pred_label = net(image.cuda())
            optim.zero_grad()
            loss = criterion(pred_label, label.cuda())
            loss.backward()
            optim.step()
            parameters = list(elem.grad for elem in net.parameters())
            gradient_list.append(parameters)
            iteration += 1

    loss_list = np.zeros(
        (GSrange * GSratio * 2 + 1,
         GSrange * GSratio * 2 + 1))

    parameter_state = np.array(list(net.parameters()))
    gradient_list = np.array(gradient_list)

    for i in range(gradient_list[0].shape[0]):  # Global shape checker
        if (gradient_list[0][i].shape != gradient_list[1][i].shape):
            print(gradient_list[0][i].shape)

    #gradient_list[0] = weight_projection(gradient_list[0])
    # gradient_list[1] =
    scalar_value = np.mean(
        weight_projection(
            gradient_list[0],
            gradient_list[1]))

    print("Scalar value : {}".format(scalar_value))

    scalar_vector = scalar_value * gradient_list[0]

    remaining_vector = gradient_list[1] - scalar_vector

    gradient_list[1] = (remaining_vector) * \
        complex_norm(gradient_list[1]) / complex_norm(remaining_vector)

    boundary = int(GSrange * GSratio)

    for x in range(-boundary, boundary + 1):
        for y in range(-boundary, boundary + 1):
            goal_param = parameter_state + x / GSratio * \
                gradient_list[0] + y / GSratio * gradient_list[1]
            goal_iteration = 0

            for param in net.parameters():
                param = goal_param[goal_iteration]
                goal_iteration += 1

            loss_list[x][y] = compute_loss(net, train_loader)

            del goal_param

            print('Done : {},{}, ratio = {}'.format(x, y, GSratio))

    iteration = 0

    for element in net.parameters():
        element = parameter_state[iteration]
        iteration += 1

    # Do not forget to reset network weight - Faire ca sur serveur
    return loss_list


def sweet_spot(net_value, net_target, mess_generator, verbose=False):
    """
    Return the according value linked to a mess generator, with a comparable loss to the net_target one
    """
    value = 1
    done = False
    nb_iter = 0
    previous_loss = float('inf')
    while(not done):
        net_value.apply(mess_generator(value))

        loss_value_list = []
        loss_target_list = []

        for i, data in enumerate(train_loader):
            image = data[0].type(torch.FloatTensor).cuda()
            label = data[1].type(torch.LongTensor).cuda()

            pred_label = net_value(image)
            pred_non_res = net_target(image)

            loss = CRITERION(pred_label, label)
            loss_non_res = CRITERION(pred_non_res, label)
            loss_value_list.append(loss.data.item())
            loss_target_list.append(loss_non_res.item())

        loss_value = np.mean(loss_value_list)
        loss_target = np.mean(loss_target_list)
        if(verbose):
            print(
                "Loss_value : {:.2f}, Loss_target : {:.2f}, messing value {}".format(
                    loss_value, loss_target, value))

        if (previous_loss < loss_value):
            print("Unstable loss evolution, cutting short")
            return value
        else:
            previous_loss = loss_value
        if(loss_value < loss_target * 4 and loss_value > loss_target * 0.9):
            done = True
            return value
        if(loss_value < loss_target):
            value = value * 2
        else:
            value = value / 2
        nb_iter = nb_iter + 1
    return value


def shooting_star(net, train_loader, ratio=1, gradient_step_range=10, LEARNING_RATE=0.01, CRITERION=nn.CrossEntropyLoss()):
    """
    Project the gradient over several steps, allowing to "see" the loss space over several iterations
    """
    new_net = copy.deepcopy(net)
    new_optim = torch.optim.SGD(new_net.parameters(), LEARNING_RATE / ratio)

    i, data = next(enumerate(train_loader))
    image = data[0].type(torch.FloatTensor).cuda()
    label = data[1].type(torch.LongTensor).cuda()

    pred_label = new_net(image)

    loss = CRITERION(pred_label, label)

    loss.backward()

    loss_list = []

    for i in range(int(ratio * gradient_step_range) + 1):
        loop_loss_list = []
        for iter_nb, data in enumerate(train_loader):
            image = data[0].type(torch.FloatTensor).cuda()
            label = data[1].type(torch.LongTensor).cuda()

            pred_label = new_net(image)
            loss = CRITERION(pred_label, label)

            loop_loss_list.append(loss.data.item())
        loss_list.append(np.mean(loop_loss_list))
        new_optim.step()
    return loss_list


def resnet_compare(
        test_net,
        model_2,
        nb_epochs=2,
        points_per_epochs=10,
        gradient_extension=False,
        gradient_ratio=1,
        gradient_range=10,
        GSgradient=False,
        GSrange=2,
        GSratio=1,
        LEARNING_RATE=0.1,
        CRITERION=nn.CrossEntropyLoss(),
        train_loader=None,
        test_loader=None,
        gradient_scale_plot=False,
        bias_inspector_report=None,
        eigenvalues_inspector=False,
        eigenvalues=None,
        eigenvectors=None,
        shift_try=False,
        shift_range=5):
    """
    Compare the training of two different networks.
    It is a toolbox set for personal research, with many options, which are referenced here.

    Arguments:
    -------------------------------------------
    test_net: The baseline network. It should be the "control point" over your experiment.

    model_2: The model which undergoes change.

    nb_epochs: Number of training epochs

    points_per_epochs: The algorithm periodicaly stops to "report" its state. Its stops this number of time per epochs.

    gradient_extension: Extend the current gradient at the current training point, and then plot the extended loss space.

    gradient_ratio: The distance between each point when the gradient_extension is enabled

    gradient_range: The maximum distance when the gradient_extension is enabled

    GSgradient: Enable the grid-search loss computation. Takes the gradient at time t and t+1, then orthonormalize them and compute the loss value over several iteration of the normalized gradient.

    GSrange: The maximum distance when the GSgradient is enabled.

    GSratio: The distance between each point when the GSgradient is enabled

    LEARNING_RATE: The usual learning rate to be taken by the SGD

    CRITERION: The loss pattern to be taken

    train_loader: The baseline train_loader to train on.

    test_loader: The baseline test_loader to test on.

    gradient_scale_plot: Plot the scale of the gradient over the course of training.

    bias_inspector_report: Plot the bias "evolution" over the course of training.
    """
    optimizer = torch.optim.SGD(test_net.parameters(), LEARNING_RATE)
    optimizer_2 = torch.optim.SGD(model_2.parameters(), LEARNING_RATE)

    score_list = []
    score_list_2 = []

    loss_list = []
    loss_list_2 = []

    if(gradient_scale_plot):
        gradient_scale_list_1 = []
        gradient_scale_list_2 = []

    if(bias_inspector_report):
        bias_diff_1 = plt_tools.bias_inspector(test_net)[0]
        bias_diff_2 = plt_tools.bias_inspector(model_2)[0]

        overall_diff_1 = [elem.cpu().detach().numpy()
                          for elem in test_net.parameters()]
        overall_diff_2 = [elem.cpu().detach().numpy()
                          for elem in model_2.parameters()]

        bias_score_evolution_1 = []
        bias_score_evolution_2 = []

        overall_score_evolution_1 = []
        overall_score_evolution_2 = []

    if(eigenvalues_inspector):
        eigenvalue_diff_list_1 = []
        eigenvalue_diff_list_2 = []

    if(gradient_extension):
        gradient_list_1 = []
        gradient_list_2 = []

    for j in range(nb_epochs):
        print("Starting epoch n°{}".format(j))

        if(gradient_extension):
            shs_range = np.linspace(
                0, gradient_range, gradient_ratio * gradient_range + 1)

            plt.figure(figsize=(10, 10))

            shooting_value_1 = shooting_star(
                test_net,
                train_loader,
                ratio=gradient_ratio,
                gradient_step_range=gradient_range,
                LEARNING_RATE=LEARNING_RATE)

            shooting_value_2 = shooting_star(
                model_2,
                train_loader,
                ratio=gradient_ratio,
                gradient_step_range=gradient_range,
                LEARNING_RATE=LEARNING_RATE)

            gradient_list_1.append(shooting_value_1)
            gradient_list_2.append(shooting_value_2)

            plt.plot(
                shs_range,
                shooting_value_1,
                label='Model1')
            plt.plot(
                shs_range,
                shooting_value_2,
                label='Model2')

            plt.xlabel(r'$\alpha$')
            plt.title("Gradient extension at the start of epoch {}".format(j))
            plt.ylabel("loss")
            plt.legend()
            plt.show()

        if(GSgradient):

            GSmap_1 = GSplot(
                test_net,
                train_loader,
                learning_rate=LEARNING_RATE,
                GSrange=GSrange,
                GSratio=GSratio)
            GSmap_2 = GSplot(
                model_2,
                train_loader,
                learning_rate=LEARNING_RATE,
                GSrange=GSrange,
                GSratio=GSratio)

            X = np.arange(-GSrange, GSrange + 1 / GSratio, 1 / GSratio)
            Y = np.arange(-GSrange, GSrange + 1 / GSratio, 1 / GSratio)
            X, Y = np.meshgrid(X, Y)

            print("Sizes: {}, {}, {}".format(X.shape, Y.shape, GSmap_1.shape))

            fig_1 = plt.figure()
            plt.title("First network")
            ax_1 = fig_1.add_subplot(111, projection='3d')
            surf_1 = ax_1.plot_surface(
                X, Y, GSmap_1, cmap=cm.get_cmap("bwr"), alpha=0.5)
            fig_1.show()

            fig_2 = plt.figure()
            plt.title("Second network")
            ax_2 = fig_2.add_subplot(111, projection='3d')
            surf_2 = ax_2.plot_surface(
                X, Y, GSmap_2, cmap=cm.get_cmap("bwr"), alpha=0.5)
            fig_2.show()

        if (j == 0 and shift_try):
            for module in model_2.modules():
                if(hasattr(module, "in_channels") and hasattr(module, "out_channels")):
                    if(module.in_channels == module.out_channels):
                        if(module.kernel_size[0] != 1):
                            torch.nn.init.uniform_(
                                module.weight, -10**-shift_range, 10**-shift_range)

        for i, data in enumerate(train_loader):
            # test_net.train()
            # model_2.train()
            image = data[0].type(torch.FloatTensor).cuda()
            label = data[1].type(torch.LongTensor).cuda()

            pred_label = test_net(image)
            pred_2 = model_2(image)

            loss = CRITERION(pred_label, label)
            loss_2 = CRITERION(pred_2, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            optimizer_2.zero_grad()
            loss_2.backward()
            optimizer_2.step()

            if(int((i - 1) * points_per_epochs / len(train_loader)) != int(i * points_per_epochs / len(train_loader)) or i == len(train_loader) - 1):

                # test_net.eval()
                # model_2.eval()

                res_score = get_score(test_net, test_loader)
                unres_score = get_score(model_2, test_loader)
                score_list.append(res_score)
                score_list_2.append(unres_score)

                loss_list.append(loss.data)
                loss_list_2.append(loss_2.data)

                if(gradient_scale_plot):
                    gradient_scale_list_1.append(get_grad_norm(test_net))
                    gradient_scale_list_2.append(get_grad_norm(model_2))

                if(eigenvalues_inspector):
                    eigenvalue_diff_list_1.append(eigenvalue_observer(
                        test_net, eigenvalues, eigenvectors))
                    eigenvalue_diff_list_2.append(
                        eigenvalue_observer(model_2, eigenvalues, eigenvectors))

                if(bias_inspector_report):  # This part causes huge GPU memory leaks

                    #init_mem_state = torch.cuda.memory_allocated()
                    #init_cached_mem_state =  torch.cuda.memory_cached()

                    bias_temp_1_inspector = plt_tools.bias_inspector(test_net)
                    bias_temp_1 = copy.deepcopy(
                        plt_tools.bias_inspector(test_net))

                    bias_temp_2_inspector = plt_tools.bias_inspector(model_2)
                    bias_temp_2 = copy.deepcopy(bias_temp_2_inspector)

                    # print("")
                    #print("Inspect: {:.2e},{:.2e}".format(float(torch.cuda.memory_allocated() - init_mem_state), float(torch.cuda.memory_cached() - init_cached_mem_state)))

                    overall_temp_1 = [elem.clone().cpu().detach().numpy()
                                      for elem in test_net.parameters()]

                    overall_temp_2 = [elem.clone().cpu().detach().numpy()
                                      for elem in model_2.parameters()]

                    #print("Overall: {:.2e},{:.2e}".format(float(torch.cuda.memory_allocated() - init_mem_state), float(torch.cuda.memory_cached() - init_cached_mem_state)))

                    bias_score_evolution_1.append(
                        compute_grad_list_diff_score(
                            bias_temp_1, bias_diff_1))
                    bias_score_evolution_2.append(
                        compute_grad_list_diff_score(
                            bias_temp_2, bias_diff_2))

                    overall_score_evolution_1.append(
                        compute_grad_list_diff_score(
                            overall_temp_1, overall_diff_1))
                    overall_score_evolution_2.append(
                        compute_grad_list_diff_score(
                            overall_temp_2, overall_diff_2))

                    #print("Evol: {:.2e},{:.2e}".format(float(torch.cuda.memory_allocated() - init_mem_state), float(torch.cuda.memory_cached() - init_cached_mem_state)))

                    bias_diff_1 = copy.deepcopy(bias_temp_1)
                    bias_diff_2 = copy.deepcopy(bias_temp_2)

                    overall_diff_1 = overall_temp_1
                    overall_diff_2 = overall_temp_2

                    #print("Copyt: {:.2e},{:.2e}".format(float(torch.cuda.memory_allocated() - init_mem_state), float(torch.cuda.memory_cached() - init_cached_mem_state)))

                    del(bias_temp_1_inspector)
                    del(bias_temp_2_inspector)

                    del(bias_temp_1)
                    del(bias_temp_2)

                    del(overall_temp_1)
                    del(overall_temp_2)

                    #print("After del before free: {:.2e},{:.2e}".format(float(torch.cuda.memory_allocated() - init_mem_state), float(torch.cuda.memory_cached() - init_cached_mem_state)))

                    torch.cuda.empty_cache()

                    #print("After free: {:.2e},{:.2e}".format(float(torch.cuda.memory_allocated() - init_mem_state), float(torch.cuda.memory_cached() - init_cached_mem_state)))

                print("Done: {}%, model1: {:.2f}%, model2: {:.2f}%, loss1: {:.4f}, loss2: {:.4f} ".format(str(int(
                    i * 100 / len(train_loader))), 100 * res_score, 100 * unres_score, loss.data, loss_2.data), end='\r')
        print("")
        torch.cuda.empty_cache()

        print("Ending epoch n°{}".format(j))

        return_list = [[score_list, score_list_2], [loss_list, loss_list_2]]

        if(gradient_scale_plot):
            return_list.append([gradient_scale_list_1, gradient_scale_list_2])

        if(bias_inspector_report):
            return_list.append(
                [bias_score_evolution_1, bias_score_evolution_2])
            return_list.append(
                [overall_score_evolution_1, overall_score_evolution_2])

        if(eigenvalues_inspector):
            return_list.append(
                [eigenvalue_diff_list_1, eigenvalue_diff_list_2])
        if(gradient_extension):
            return_list.append(
                [gradient_list_1, gradient_list_2])
    return return_list


def train_and_test(
        depth,
        width,
        nb_epochs,
        points_per_epochs=10,
        messed_up_init_generator=None,
        gradient_extension=False,
        gradient_ratio=1,
        gradient_range=10,
        statistics=False,
        detector_plot=False,
        biased_start=False,
        zero_start=False):
    net = GenResNet(depth, width).cuda()
    non_resNet = GenResNet(depth, width, residual=False).cuda()

    if(zero_start):
        net.apply(zero_mess)
        non_resNet.apply(zero_mess)

    if not(messed_up_init_generator is None):
        optimal_mess = sweet_spot(
            net,
            non_resNet,
            messed_up_init_generator,
            verbose=True)
        print("Final mess : {:2f}".format(optimal_mess))
        # replace with optimal_mess
        net.apply(messed_up_init_generator(optimal_mess))

    if(biased_start):
        for name, param in non_resNet.named_parameters():
            if('.conv.weight' in name):
                in_c, out_c, kernel, kernel = param.size()
                for i in range(in_c):
                    for j in range(out_c):
                        torch.nn.init.normal_(
                            param[i][j][1][1], 1 / in_c, 1 / (2 * in_c))

    optimizer = torch.optim.SGD(net.parameters(), LEARNING_RATE)
    optimizer_non_res = torch.optim.SGD(non_resNet.parameters(), LEARNING_RATE)

    score_list = []
    score_list_non_res = []

    loss_list = []
    loss_list_non_res = []

    if(statistics):
        stat = []
        stat_unres = []

    if(detector_plot):
        detector_done = False

    for j in range(nb_epochs):
        print("Starting epoch n°{}".format(j))

        if(gradient_extension):
            shs_range = np.linspace(
                0, gradient_range, gradient_ratio * gradient_range + 1)

            plt.figure(figsize=(10, 10))
            plt.plot(
                shs_range,
                shooting_star(
                    net,
                    train_loader,
                    ratio=gradient_ratio,
                    gradient_step_range=gradient_range),
                label='Residual')
            plt.plot(
                shs_range,
                shooting_star(
                    non_resNet,
                    train_loader,
                    ratio=gradient_ratio,
                    gradient_step_range=gradient_range),
                label='NonRes')
            plt.xlabel(r'$\alpha$')
            plt.title("Gradient extension at the start of epoch {}".format(j))
            plt.yscale('log')
            plt.ylabel("log(loss)")
            plt.legend()
            plt.show()

        for i, data in enumerate(train_loader):
            image = data[0].type(torch.FloatTensor).cuda()
            label = data[1].type(torch.LongTensor).cuda()

            pred_label = net(image)
            pred_non_res = non_resNet(image)

            loss = CRITERION(pred_label, label)
            loss_non_res = CRITERION(pred_non_res, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            optimizer_non_res.zero_grad()
            loss_non_res.backward()
            optimizer_non_res.step()

            if(int((i - 1) * points_per_epochs / len(train_loader)) != int(i * points_per_epochs / len(train_loader)) or i == len(train_loader) - 1):
                res_score = get_score(net, test_loader)
                unres_score = get_score(non_resNet, test_loader)
                score_list.append(res_score)
                score_list_non_res.append(unres_score)

                loss_list.append(loss.data)
                loss_list_non_res.append(loss_non_res.data)

                if(statistics):
                    stat.append(get_gradient_statistics(net))
                    stat_unres.append(get_gradient_statistics(non_resNet))

                print("Done: {}%, residual: {:.2f}%, non_res: {:.2f}%, loss: {:.4f}, loss nonres: {:.4f} ".format(str(int(
                    i * 100 / len(train_loader))), 100 * res_score, 100 * unres_score, loss.data, loss_non_res.data), end='\r')

                if(detector_plot and not detector_done):
                    if(loss_non_res.data.item() < 2.2):
                        weight_reporter(non_resNet)
                        detector_done = True
        print("")

        print("Ending epoch n°{}".format(j))

    if (statistics):
        return((score_list, loss_list), (score_list_non_res, loss_list_non_res), (stat, stat_unres))
    return (score_list, loss_list), (score_list_non_res, loss_list_non_res)


def plot_and_compare(
        max_depth,
        width,
        nb_epochs=5,
        points_per_epochs=10,
        messed_up_init_generator=None,
        skip_zero_depth=True):

    plt.style.use("ggplot")

    score_range = []
    score_range_non_res = []

    loss_range = []
    loss_range_non_res = []

    epoch_range = np.linspace(0, nb_epochs, points_per_epochs * nb_epochs)

    if(skip_zero_depth):
        init_depth = 1
    else:
        init_depth = 0

    for i in range(init_depth, max_depth):
        score_and_loss, score_and_loss_unres = train_and_test(
            i, width, nb_epochs, points_per_epochs, messed_up_init_generator=messed_up_init_generator)
        curr_score, curr_loss = score_and_loss
        curr_score_unres, curr_loss_unres = score_and_loss_unres
        plt.figure(figsize=(10, 10))
        plt.plot(epoch_range, curr_score, label="Residual")
        plt.plot(epoch_range, curr_score_unres, label="NonRes")
        plt.title("Score with width :{} and depth: {}".format(width, i))
        plt.xlabel("Epochs")
        plt.ylabel("Score")
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 10))
        plt.plot(epoch_range, curr_loss, label="Residual")
        plt.plot(epoch_range, curr_loss_unres, label="NonRes")
        plt.title("Loss with width :{} and depth: {}".format(width, i))
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

        score_range.append(curr_score)
        score_range_non_res.append(curr_score_unres)
        loss_range.append(curr_loss)
        loss_range_non_res.append(curr_loss_unres)

    return (score_range, loss_range), (score_range_non_res, loss_range_non_res)


def get_gradient_statistics(net):
    return [(torch.mean(param.grad).item(), torch.var(param.grad).item())
            for param in net.parameters()]


def net_training(
        net,
        train_loader,
        test_loader,
        learning_rate,
        variance_modifier,
        nb_epoch,
        CRITERION=nn.CrossEntropyLoss()):
    curr_net = copy.deepcopy(net).cuda()
    optimizer = torch.optim.SGD(curr_net.parameters(), learning_rate)

    for module in curr_net.modules():
        if(hasattr(module, "in_channels") and hasattr(module, "out_channels")):
            if(module.in_channels == module.out_channels):
                if(module.kernel_size[0] != 1):
                    messing.var_modifier(module.weight, variance_modifier)

    for epoch in range(nb_epoch):
        for i, data in enumerate(train_loader):
            image = data[0].type(torch.FloatTensor).cuda()
            label = data[1].type(torch.LongTensor).cuda()

            pred = curr_net(image)

            loss = CRITERION(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return get_score(curr_net, test_loader)


def lr_variance_gradient_norm(
        net,
        train_loader,
        test_loader,
        learning_rate,
        variance_modifier,
        nb_epoch,
        CRITERION=nn.CrossEntropyLoss()):

    curr_net = copy.deepcopy(net).cuda()
    optimizer = torch.optim.SGD(curr_net.parameters(), learning_rate)

    for module in curr_net.modules():
        if(hasattr(module, "in_channels") and hasattr(module, "out_channels")):
            if(module.in_channels == module.out_channels):
                if(module.kernel_size[0] != 1):
                    messing.var_modifier(module.weight, variance_modifier)

    for i, data in enumerate(train_loader):
        image = data[0].type(torch.FloatTensor).cuda()
        label = data[1].type(torch.LongTensor).cuda()

        pred = curr_net(image)

        loss = CRITERION(pred, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        break

    return get_grad_norm(curr_net)


def lr_variance_GD_strategy(
        base_net,
        var_range,
        lr_range,
        train_loader,
        test_loader,
        epoch_limit=1,
        gradient_norm_computation=False):
    return_matrix = np.zeros((len(var_range), len(lr_range)))
    if(gradient_norm_computation):
        gradient_matrix = np.zeros_like(return_matrix)
    for i in range(len(var_range)):
        var_value = var_range[i]

        for j in range(len(lr_range)):
            lr_value = lr_range[j]

            test_score = net_training(
                base_net,
                train_loader=train_loader,
                test_loader=test_loader,
                learning_rate=lr_value,
                variance_modifier=var_value,
                nb_epoch=epoch_limit)
            return_matrix[i][j] = test_score

            if(gradient_norm_computation):
                gradient_matrix[i][j] = lr_variance_gradient_norm(base_net,
                                                                  train_loader=train_loader,
                                                                  test_loader=test_loader,
                                                                  learning_rate=lr_value,
                                                                  variance_modifier=var_value,
                                                                  nb_epoch=epoch_limit)

            if(gradient_norm_computation):
                print('Done: var:{}, lr{} - Score = {}, gradient norm: {}'.format(
                    var_value, lr_value, test_score, gradient_matrix[i][j]))
            else:
                print('Done: var:{}, lr{} - Score = {}'.format(var_value,
                                                               lr_value, test_score))
    return_list = []
    return_list.append(return_matrix)

    if(gradient_norm_computation):
        return_list.append(gradient_matrix)
    return return_list


def eigenvalue_observer(network, eigenvalues, eigenvectors):
    eigenvalue_diff_list = []
    for i in range(eigenvectors.shape[0]//20):
        torch_eigen = torch.Tensor(
            eigenvectors[i*20: min((i+1)*20, eigenvectors.shape[0])]).cuda()
        eigenvalue_diff_list.append(np.linalg.norm(network.without_eigen(torch_eigen).detach(
        ).cpu().numpy() - network.with_eigen(torch_eigen).detach().cpu().numpy()))
    return eigenvalue_diff_list


def deactivate_batchnorm(m):
    if isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()
        m.eval()
        with torch.no_grad():
            m.weight.fill_(1.0)
            m.bias.zero_()
