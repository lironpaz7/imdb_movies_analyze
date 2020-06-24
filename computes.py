def compute_precision(real, predicted):
    """
    :param real: Vector of the real label.
    :param predicted: Vector of the predicted label.
    :return: precision value.
    """
    tp, fp = 0, 0
    for x, y in zip(real, predicted):
        if x == y and x == 1:
            tp += 1
        elif x != y and x == 0:
            fp += 1
    return tp / (fp + tp)


def compute_recall(real, predicted):
    """
    :param real: Vector of the real label.
    :param predicted: Vector of the predicted label.
    :return: recall value.
    """
    tp, fn = 0, 0
    for x, y in zip(real, predicted):
        if x == y and x == 1:
            tp += 1
        elif x != y and x == 1:
            fn += 1
    return tp / (tp + fn)


def compute_accuracy(real, predicted):
    """
    :param real: Vector of the real label.
    :param predicted: Vector of the predicted label.
    :return: accuracy value.
    """
    return sum([x == y for x, y in zip(real, predicted)]) / len(real)
