import numpy as np
from sklearn import metrics


def compute_auc_score(pred, label, return_threshold=False):
    """
    It computes the AUC score of a binary classification problem

    :param pred: the output of the model
    :param label: the ground truth label
    :param return_threshold: If True, returns the threshold used to compute the AUC score, defaults to False (optional)
    :return: The AUC score and the threshold
    """
    label = label.reshape(-1).cpu().detach().numpy()
    label = (label >= label.mean()).astype(np.uint8).tolist()
    pred = pred.reshape(-1).cpu().detach().numpy().astype(float).tolist()
    fpr, tpr, threshold = metrics.roc_curve(label, pred)
    auc = metrics.auc(fpr, tpr)
    if return_threshold:
        return auc, threshold
    else:
        return auc
