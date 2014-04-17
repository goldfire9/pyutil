"""Utilities to evaluate the predictive performance of models

Functions named as *_score return a scalar value to maximize: the higher the
better

Function named as *_loss return a scalar value to minimize: the lower the
better
"""

# Authors: Alexandre Gramfort <alexandre.gramfort@inria.fr>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Olivier Grisel <olivier.grisel@ensta.org>
# License: BSD Style.

import numpy as np

def precision_recall_curve(y_true, y_score, thresholds=None, get_count=False):
    """compute Precision Recall Curve (PRC)

    Note: this implementation is restricted to the binary classification task.

    Parameters
    ----------

    y_true : array, shape = [n_samples]
        true binary labels

    y_score : array, shape = [n_samples]
        target scores, can either be probability estimates of
        the positive class, confidence values, or binary decisions.

    Returns
    -------
    fpr : array, shape = [>2]
        False Positive Rates

    tpr : array, shape = [>2]
        True Positive Rates

    thresholds : array, shape = [>2]
        Thresholds on y_score used to compute fpr and tpr.

        *Note*: Since the thresholds are sorted from low to high values,
        they are reversed upon returning them to ensure they
        correspond to both fpr and tpr, which are sorted in reversed order
        during their calculation.


    Examples
    --------
    >>> import numpy as np
    >>> from sklearn import metrics
    >>> y = np.array([1, 1, 2, 2])
    >>> scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> fpr, tpr, thresholds = metrics.roc_curve(y, scores)
    >>> fpr
    array([ 0. ,  0.5,  0.5,  1. ])

    References
    ----------
    http://en.wikipedia.org/wiki/Receiver_operating_characteristic

    """
    y_true = np.ravel(y_true)
    classes = np.unique(y_true)

    # PRC only for binary classification
    if classes.shape[0] != 2:
        raise ValueError("ROC is defined for binary classification only")

    y_score = np.ravel(y_score)

    n_pos = float(np.sum(y_true == classes[1]))  # nb of true positive
    n_neg = float(np.sum(y_true == classes[0]))  # nb of true negative

    if not thresholds:
        thresholds = np.unique(y_score)
        
    neg_value, pos_value = classes[0], classes[1]

    recall = np.empty(thresholds.size, dtype=np.float)  # True positive rate
    precis = np.empty(thresholds.size, dtype=np.float)  # 
    pred_hit =  np.empty(thresholds.size, dtype=np.int)
    pred_pos =  np.empty(thresholds.size, dtype=np.int)

    # Build tpr/fpr vector
    current_pos_count = current_neg_count = current_pos_pred = sum_pos = sum_neg = sum_pos_pred = idx = 0

    signal = np.c_[y_score, y_true]
    sorted_signal = signal[signal[:, 0].argsort(), :][::-1]
    last_score = sorted_signal[0][0]
    for score, value in sorted_signal:
        if score == last_score:
            current_pos_pred += 1
            if value == pos_value:
                current_pos_count += 1
            else:
                current_neg_count += 1
        else:
            sum_pos_pred += current_pos_pred
            sum_pos += current_pos_count
            sum_neg += current_neg_count
            recall[idx] = (sum_pos) / n_pos
            precis[idx] = (sum_pos) / float(sum_pos_pred)
            pred_hit[idx] = sum_pos
            pred_pos[idx] = sum_pos_pred
            current_pos_count = 1 if value == pos_value else 0
            current_neg_count = 1 if value == neg_value else 0
            current_pos_pred = 1
            idx += 1
            last_score = score
    else:
        recall[-1] = (sum_pos + current_pos_count) / n_pos
        precis[-1] = (sum_neg + current_pos_count) / (sum_pos_pred + current_pos_pred)
        pred_hit[-1] = sum_pos + current_pos_count
        pred_pos[-1] = sum_pos_pred + current_pos_pred

    # hard decisions, add (0,0)
    if precis.shape[0] == 2:
        precis = np.array([0.0, precis[0], precis[1]])
        recall = np.array([0.0, recall[0], recall[1]])
    # trivial decisions, add (0,0) and (1,1)
    elif precis.shape[0] == 1:
        precis = np.array([0.0, precis[0], 1.0])
        recall = np.array([0.0, recall[0], 1.0])

    if get_count:
        return pred_hit, pred_pos, precis, recall, thresholds[::-1]
    else:
        return precis, recall, thresholds[::-1]

