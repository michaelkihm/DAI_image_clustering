import numpy as np 
from tensorflow import keras 
from scipy.optimize import linear_sum_assignment

def unsupervised_labels(y, yp, n_classes:int, n_clusters:int):
    """
    @brief Linear assignment algorithm
    @param (tensor): Ground truth labels
    @param  yp (tensor): Predicted clusters
    @param n_classes: Number of classes
    @param n_clusters: Number of clusters
    """
    assert n_classes == n_clusters

    # initialize count matrix
    C = np.zeros([n_clusters, n_classes])

    # populate count matrix
    for i in range(len(y)):
        C[int(yp[i]), int(y[i])] += 1

    # optimal permutation using Hungarian Algo
    # the higher the count, the lower the cost
    # so we use -C for linear assignment
    row, col = linear_sum_assignment(-C)

    # compute accuracy
    accuracy = C[row, col].sum() / C.sum()

    return accuracy * 100


def lr_schedule(epoch):
    """
    @brief Simple learning rate scheduler
    @param epoch Which epoch 
    """
    lr = 1e-3
    power = epoch // 400
    lr *= 0.8**power

    return lr


class AccuracyCallback(keras.callbacks.Callback):
    """
    @brief Callback to compute the accuracy every epoch by
        calling the eval() method.
    @param net (Model): Object with a network model to evaluate. 
            Must support the eval() method.
    """
    def __init__(self, net):
        super(AccuracyCallback, self).__init__()
        self.net = net 
    #pylint: disable=unused-argument
    def on_epoch_end(self, epoch, logs=None):
        self.net.eval()