import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve

from src.utilities import flatten


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])


def in_top_k(y_true, y_pred, k=5):
    """
    Trims the y_pred to a length of k on the right and calculates the accuracies
    to the y_true.
    :y_true: list of actual values to be predicted (list)
    :y_pred: list of predicted values (ordered by propensity) (list)
    :k: number of predictions to consider (int)
    :return: a list of top_k accuracies (list)
    """
    y_pred = map(lambda x: x[0:k], y_pred)
    joined = zip(y_true, y_pred)
    accuracies_at_k = map(lambda x: np.mean([s in x[1] for s in x[0]]), joined)
    return accuracies_at_k


def top_k_categorical_accuracy(y_true, y_pred, k=5):
    """
    Trims the y_pred to a length of k by the right and calculates the topK categorical accuracy to the y_true.
    :y_true: list of actual values to be predicted (list)
    :y_pred: list of predicted values (ordered by propensity) (list)
    :k: number of predictions to consider (int)
    :return: the average of the accuracies (float)
    """
    return np.mean(in_top_k(y_true, y_pred, k=k))


def top_k_hit_ratio(y_true, y_pred, k=5):
    """
    Trims the y_pred to a length of k on the right and calculates the hit ratio at k
    :y_true: list of actual values to be predicted (list)
    :y_pred: list of predicted values (ordered by propensity) (list)
    :k: number of predictions to consider (int)
    :return: the average of the git ratios(float)
    """
    return np.mean(map(lambda x: x > 0, in_top_k(y_true, y_pred, k=k)))


def rank_precision_recall_fscore_support_at_k(y_true, y_pred, k=5):
    """
    Trims the y_pred to a length of k by the right and calculates the average
    accuracies to the y_true.
    :y_true: list of actual values to be predicted (list)
    :y_pred: list of predicted values (ordered by propensity) (list)
    :k: number of predictions to consider (int)
    :return: the average of the accuracies (float)
    """
    precision, recall, fscore, support = [], [], [], []
    categories = set(flatten(y_true))
    for category in categories:
        y_x = [category in x for x in y_true]
        l_x = [category in x[0:k] for x in y_pred]
        precision_, recall_, fscore_, support_ = np.array(precision_recall_fscore_support(y_true=y_x, y_pred=l_x))[:, 1]
        precision.append(precision_)
        recall.append(recall_)
        fscore.append(fscore_)
        support.append(support_)
    return precision, recall, fscore, support, categories


def generate_rank_reports(y_true, y_pred, k_range=None):
    """
    Given the true values and the predicted ones, it generates a dataframe containing 
    the map@k, the topKcategoricalAccuracy and the hitsRatio@K, the precision and recall
    @k by product.
    :y_true: list of actual values to be predicted (list)
    :y_pred: list of predicted values (ordered by propensity) (list)
    :k_range: range number of predictions to consider. If not specified, it will be
    considered as the whole range, by counting the total number of unique labels
    by using y_true (int|None)
    :return: a table with all the metrics for all the K values (pd.Dataframe)
    """
    k_range = [x + 1 for x in range(len(set(flatten(y_true))))] if type(k_range) == type(None) else k_range
    # Compute general metrics (dependant of all the leads generated)
    _map, _acc, _hit = [], [], []
    for k in k_range:
        _map.append(mapk(y_true, y_pred, k))
        _acc.append(top_k_categorical_accuracy(y_true, y_pred, k))
        _hit.append(top_k_hit_ratio(y_true, y_pred, k))
    metrics_at_k = pd.DataFrame({"k": k_range,
                                 "Map@k": _map,
                                 "TopAcc@k": _acc,
                                 "TopHit@k": _hit})[["k", "Map@k", "TopAcc@k", "TopHit@k"]]

    # Compute product metrics (based on each of the products performance)
    product, k, segment, precision, recall, fscore, support = [], [], [], [], [], [], []
    for k_ in k_range:
        precision_, recall_, fscore_, support_, products_ = rank_precision_recall_fscore_support_at_k(y_true=y_true,
                                                                                                      y_pred=y_pred,
                                                                                                      k=k_)
        precision.extend(precision_)
        recall.extend(recall_)
        fscore.extend(fscore_)
        support.extend(support_)
        k.extend([k_] * len(products_))
        product.extend(products_)

    product_metrics_at_k = pd.DataFrame({"product": product,
                                         "k": k,
                                         "precision": precision,
                                         "recall": recall,
                                         "fscore": fscore,
                                         "support": support})[["product", "k", "precision",
                                                               "recall", "fscore", "support"]]
    return metrics_at_k, product_metrics_at_k


def mae(truth, preds):
    """
    Calculates the Mean average error
    :param truth: list of actual values to be predicted (list)
    :param preds: list of predicted values (ordered by propensity) (list)
    :return: the mean absolute error (float)
    """
    return np.abs(truth - preds).mean()


def generate_binary_reports(y_true, y_pred, path, alias="", uplift_bins=100):
    """
    Given a target variable and a set of predictions, calculates a set of standard metrics to measure the performance
    :param y_true: list of actual values to be predicted (list)
    :param y_pred: list of predicted values (ordered by propensity) (list)
    :param path: path to the folder where the reports must be saved (str|unicode)
    :param alias: alias of the reports saved. This will be appended in the filenames of the reports (str|unicode)
    :return: None (void)
    """
    # ROC
    fig = plt.figure(figsize=[10, 8])
    ax = fig.add_subplot(1, 1, 1)

    fpr, tpr, _ = roc_curve(y_true=y_true,
                            y_score=y_pred)
    gini = -1 + 2 * roc_auc_score(y_true=y_true,
                                  y_score=y_pred)
    plt.plot(fpr, tpr, color="navy",
             lw=2, label="ROC curve (gini = %0.3f)" % (gini))
    plt.plot([0, 1], [0, 1], color="darkorange", lw=0.5, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True_positive_rate")
    plt.title("Receiver Operating Characteristic")
    major_ticks = np.arange(0, 1.05, 0.25)
    minor_ticks = np.arange(0, 1.05, 0.05)
    ax.set_xticks(major_ticks)
    ax.set_xticks(minor_ticks, minor=True)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    plt.grid(which="minor", alpha=0.3)
    plt.grid(which="major", alpha=0.5)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(path, "ROC_{}.jpg".format(alias)))
    plt.close()

    # Uplift math
    df_results = pd.DataFrame({"PRED": y_pred, "TRUE": y_true})
    df_results["BIN"] = pd.cut(df_results.PRED,
                               bins=np.percentile(df_results.PRED, np.arange(0, 100.0000000000001, 100.0 / uplift_bins)),
                               labels=list(reversed(np.arange(0, 100.0000000000001, 100.0 / uplift_bins)[1:])))

    props = df_results.groupby("BIN").TRUE.mean()
    sums = df_results.groupby("BIN").TRUE.sum()
    sizes = df_results.groupby("BIN").TRUE.count()

    df_uplift = (pd.DataFrame({"Total 1s in target": sums,
                               "Population size": sizes,
                               "Proportion 1s in target": props,
                               "Marginal uplift": props / df_results.TRUE.mean()})
                 .reset_index()
                 .sort_values(by="BIN", ascending=False)
                 .reset_index(drop=True))

    df_uplift["Cumulative proportion 1s in target"] = df_uplift["Total 1s in target"].cumsum() / df_uplift[
        "Population size"].cumsum()
    df_uplift["Cumulative uplift"] = df_uplift["Cumulative proportion 1s in target"] / df_results.TRUE.mean()
    df_uplift.to_excel(os.path.join(path, "uplift_{}.xls".format(alias)))

    # Uplift net    
    fig = plt.figure(figsize=[13, 8])
    ax = fig.add_subplot(1, 1, 1)
    plt.bar(df_uplift["BIN"], df_uplift["Proportion 1s in target"], color="navy", label="Net Uplift")

    plt.ylim([0, 1])
    plt.xlim([-5, 105])
    plt.title("Uplift Curve")
    plt.legend(loc="upper right")
    major_ticks_y = np.arange(0, 1.05, 0.25)
    minor_ticks_y = np.arange(0, 1.05, 0.05)
    major_ticks_x = np.arange(0, 105, 25)
    minor_ticks_x = np.arange(0, 105, 5)
    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    plt.axhline(df_results.TRUE.mean(), color="darkorange")
    plt.grid(which="minor", alpha=0.3)
    plt.grid(which="major", alpha=0.5)
    plt.savefig(os.path.join(path, "net_uplift_{}.jpg".format(alias)))
    plt.close()

    # Uplift cumulative                       
    fig = plt.figure(figsize=[13, 8])
    ax = fig.add_subplot(1, 1, 1)
    plt.bar(df_uplift["BIN"], df_uplift["Cumulative proportion 1s in target"], color="navy", label="Cumulative Uplift")

    plt.ylim([0, 1])
    plt.xlim([-5, 105])
    plt.title("Uplift Curve")
    plt.legend(loc="upper right")
    major_ticks_y = np.arange(0, 1.05, 0.25)
    minor_ticks_y = np.arange(0, 1.05, 0.05)
    major_ticks_x = np.arange(0, 105, 25)
    minor_ticks_x = np.arange(0, 105, 5)
    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
    plt.axhline(df_results.TRUE.mean(), color="darkorange")
    plt.grid(which="minor", alpha=0.3)
    plt.grid(which="major", alpha=0.5)
    plt.savefig(os.path.join(path, "cumulative_uplift_{}.jpg".format(alias)))
    plt.close()
