import numpy as np

def ranking_metric(evalpred):
    scores0 = evalpred[0][0]
    scores1 = evalpred[0][1]
    labels = evalpred[1]

    # labels:
    # 0 or 1: word 0 or 1 is more legible, other unknown
    # 2: both words are equally legible
    # 3: neither word is legible

    pairs_evaluated = 0
    pairs_correct = 0
    scores0 = 1 / (1 + np.exp(-scores0))
    scores1 = 1 / (1 + np.exp(-scores1))
    for i in range(scores0.shape[0]):
        if labels[i] < 2:
            pairs_evaluated += 1
            if labels[i] == 0:
                if scores0[i] >= scores1[i]:
                    pairs_correct += 1
            elif labels[i] == 1:
                if scores1[i] >= scores0[i]:
                    pairs_correct += 1

        accuracy = pairs_correct / pairs_evaluated
    return {'accuracy': accuracy}


def binary_classification_metric(evalpred):
    scores0 = evalpred[0][0]
    scores1 = evalpred[0][1]
    labels = evalpred[1]

    # labels:
    # 0 or 1: word 0 or 1 is more legible, other unknown
    # 2: both words are equally legible
    # 3: neither word is legible

    words_evaluated = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0
    scores0 = 1 / (1 + np.exp(-scores0))
    scores1 = 1 / (1 + np.exp(-scores1))
    for i in range(scores0.shape[0]):
        if labels[i] < 2:
            words_evaluated += 1
        else:
            words_evaluated += 2
        if labels[i] == 0:
            if scores0[i] > 0.5:
                true_positives += 1
            else:
                false_negatives += 1
        elif labels[i] == 1:
            if scores1[i] > 0.5:
                true_positives += 1
            else:
                false_negatives += 1
        elif labels[i] == 2:
            if scores0[i] > 0.5:
                true_positives += 1
            else:
                false_negatives += 1
            if scores1[i] > 0.5:
                true_positives += 1
            else:
                false_negatives += 1
        elif labels[i] == 3:
            if scores0[i] < 0.5:
                true_negatives += 1
            else:
                false_positives += 1
            if scores1[i] < 0.5:
                true_negatives += 1
            else:
                false_positives += 1

        # calculate precision, recall, accuracy and f1 score
        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (true_positives + false_negatives + 1e-6)
        accuracy = (true_positives + true_negatives) / (words_evaluated + 1e-6)
        f1_score = 2 * precision * recall / (precision + recall + 1e-6)
    return {'precision': precision, 'recall': recall, 'accuracy': accuracy, 'f1_score': f1_score}
