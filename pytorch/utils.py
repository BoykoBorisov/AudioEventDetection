import torch
import numpy as np
from sklearn import metrics
from scipy import stats
import os
import wave

def move_data_to_device(x, device):
    if 'float' in str(x.dtype):
        x = torch.Tensor(x)
    elif 'int' in str(x.dtype):
        x = torch.LongTensor(x)
    else:
        return x

    return x.to(device)


def get_audiofile_name_from_audioset_csv_row(row):
    return row[0].strip(",") + "_" + str(row[1].strip().strip(",")) + ".wav"


def remove_corrupt_audiofiles(filedir):
    print("Removing corrupt audiofiles...")
    error_count = 0
    full_count = 0
    for file_name in os.listdir(filedir):
      try:
        full_count += 1
        file = wave.open(os.path.join(filedir, file_name))
        file.close()
      except:
        error_count += 1
        print(file_name, error_count, "/", full_count)
        os.remove(os.path.join(filedir, file_name))
    print("Process done, removed files count: " + error_count)

"""
    y_hat: numpy array (num_samples, num_classes), model prediction
    y: numpy array (num_samples, num_classes), ground truth
    return: dict with MAP, AUC, d-prime
"""
def get_stats(y_hat, y):
    num_classes = y_hat.shape[1]  
    ap_per_class = np.zeros(num_classes)
    auc_per_class = np.zeros(num_classes)
    threshold_per_class = np.zeros(num_classes)
    # print(y[:,0])
    for cls in range(num_classes):
        ap_per_class[cls] = metrics.average_precision_score(y[:,cls], y_hat[:,cls], average=None)
        auc_per_class[cls] = metrics.roc_auc_score(y[:, cls], y_hat[:, cls], average=None)
        precision, recall, threshold =  metrics.precision_recall_curve(y[:, cls], y_hat[:, cls])
        fscore = (4 * precision * recall) / (5 * precision + recall + 1)
        threshold_per_class[cls] = threshold[np.argmax(fscore)]

    mean_ap = np.mean(ap_per_class)
    mean_auc = np.mean(auc_per_class)
    d_prime = stats.norm().ppf(mean_auc) * np.sqrt(2.0)
    return {"MAP": mean_ap, "AUC": mean_auc, "d-prime": d_prime, "class_ap": ap_per_class, "thresholds": threshold_per_class}