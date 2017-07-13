# DQN, Double DQN, A3C, action conditional video predictions, AlphaGo
import sys
import numpy as np
from collections import Counter
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import coverage_error, label_ranking_loss, hamming_loss, accuracy_score

def patk(predictions, labels):
    pak = np.zeros(3)
    K = np.array([1, 3, 5])
    for i in range(predictions.shape[0]):
        pos = np.argsort(-predictions[i, :])
        y = labels[i, :]
    y = y[pos]
    for j in range(3):
        k = K[j]
        pak[j] += (np.sum(y[:k]) / k)
    pak = pak / predictions.shape[0]
    return pak * 100.

'''
def precision_at_k(predictions, labels, k):
    act_set = 
'''
def cm_precision_recall(prediction,truth):
  """Evaluate confusion matrix, precision and recall for given set of labels and predictions
     Args
       prediction: a vector with predictions
       truth: a vector with class labels
     Returns:
       cm: confusion matrix
       precision: precision score
       recall: recall score"""
  confusion_matrix = Counter()

  positives = [1]

  binary_truth = [x in positives for x in truth]
  binary_prediction = [x in positives for x in prediction]

  for t, p in zip(binary_truth, binary_prediction):
    confusion_matrix[t,p] += 1

  cm = np.array([confusion_matrix[True,True], confusion_matrix[False,False], confusion_matrix[False,True], confusion_matrix[True,False]])
  #print cm
  precision = (cm[0]/(cm[0]+cm[2]+0.000001))
  recall = (cm[0]/(cm[0]+cm[3]+0.000001))
  return cm, precision, recall

def bipartition_scores(labels,predictions):
    """ Computes bipartitation metrics for a given multilabel predictions and labels
      Args:
        logits: Logits tensor, float - [batch_size, NUM_LABELS].
        labels: Labels tensor, int32 - [batch_size, NUM_LABELS].
      Returns:
        bipartiation: an array with micro_precision, micro_recall, micro_f1,macro_precision, macro_recall, macro_f1"""
    sum_cm=np.zeros((4))
    macro_precision=0
    macro_recall=0
    for i in range(labels.shape[1]):
        truth=labels[:,i]
        prediction=predictions[:,i]
        cm,precision,recall=cm_precision_recall(prediction, truth)
        sum_cm+=cm
        macro_precision+=precision
        macro_recall+=recall
    
    macro_precision=macro_precision/labels.shape[1]
    macro_recall=macro_recall/labels.shape[1]
    macro_f1 = 2*(macro_precision)*(macro_recall)/(macro_precision+macro_recall+0.000001)
    
    micro_precision = sum_cm[0]/(sum_cm[0]+sum_cm[2]+0.000001)
    micro_recall=sum_cm[0]/(sum_cm[0]+sum_cm[3]+0.000001)
    micro_f1 = 2*(micro_precision)*(micro_recall)/(micro_precision+micro_recall+0.000001)
    bipartiation = np.asarray([micro_precision, micro_recall, micro_f1,macro_precision, macro_recall, macro_f1])
    return bipartiation


def evaluate(predictions, labels, threshold=0.4, multi_label=True):
    '''
        True Positive  :  Label : 1, Prediction : 1
        False Positive :  Label : 0, Prediction : 1
        False Negative :  Label : 0, Prediction : 0
        True Negative  :  Label : 1, Prediction : 0
        Precision      :  TP/(TP + FP)
        Recall         :  TP/(TP + FN)
        F Score        :  2.P.R/(P + R)
        Ranking Loss   :  The average number of label pairs that are incorrectly ordered given predictions
        Hammming Loss  :  The fraction of labels that are incorrectly predicted. (Hamming Distance between predictions and labels)
    '''
    assert predictions.shape == labels.shape, "Shapes: %s, %s" % (predictions.shape, labels.shape,)
    metrics = dict()
    if not multi_label:
        metrics['bae'] = BAE(labels, predictions)
        labels, predictions = np.argmax(labels, axis=1), np.argmax(predictions, axis=1)

        metrics['accuracy'] = accuracy_score(labels, predictions)
        metrics['micro_precision'], metrics['micro_recall'], metrics['micro_f1'], _ = \
            precision_recall_fscore_support(labels, predictions, average='micro')
        metrics['macro_precision'], metrics['macro_recall'], metrics['macro_f1'], metrics['coverage'], \
            metrics['average_precision'], metrics['ranking_loss'], metrics['pak'], metrics['hamming_loss'] \
            = 0, 0, 0, 0, 0, 0, 0, 0

    else:
        metrics['coverage'] = coverage_error(labels, predictions)
        metrics['average_precision'] = label_ranking_average_precision_score(labels, predictions)
        metrics['ranking_loss'] = label_ranking_loss(labels, predictions)
        
        for i in range(predictions.shape[0]):
            predictions[i, :][predictions[i, :] >= threshold] = 1
            predictions[i, :][predictions[i, :] < threshold] = 0

        metrics['bae'] = 0
        metrics['patk'] = patk(predictions, labels)
        metrics['micro_precision'], metrics['micro_recall'], metrics['micro_f1'], metrics['macro_precision'], \
            metrics['macro_recall'], metrics['macro_f1'] = bipartition_scores(labels, predictions)
    return metrics
