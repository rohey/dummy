import torch
import matplotlib.pyplot as plt


if __name__ == '__main__':

    folder = '12_11_2020'

    for i in range(3):
        name = folder + '/efficient_net_' + str(i) + '.torch'
        state_dict, training_history, validation_history = torch.load(name)
        # plt.plot(training_history)
        plt.plot(validation_history[50:])



    for i in range(3):
        name = folder + '/simple_net_' + str(i) + '.torch'
        state_dict, training_history, validation_history = torch.load(name)
        # plt.plot(training_history)
        plt.plot(validation_history[50:])

    plt.yscale("log")
    plt.show()