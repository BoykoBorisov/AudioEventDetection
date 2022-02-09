import csv
import os
import numpy as np
from torch.utils.data import Dataset
import librosa
import random
from utils import get_audiofile_name_from_audioset_csv_row

def get_id_to_labels():
  id_to_label = {}
  with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets/Audioset/class_labels_indices.csv"))) as file:
    reader = csv.reader(file, delimiter=",")
    rows = list(reader)
    for i in range(1, len(rows)):
      id_to_label[rows[i][1]] = rows[i][2]
    return id_to_label

def get_index_to_id():
  index_to_id = {}
  with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets/Audioset/class_labels_indices.csv"))) as file:
    reader = csv.reader(file, delimiter=",")
    rows = list(reader)
    for i in range(1, len(rows)):
      index_to_id[int(rows[i][0])] = rows[i][1]
    return index_to_id

def get_id_to_index():
  id_to_index = {}
  with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets/Audioset/class_labels_indices.csv"))) as file:
    reader = csv.reader(file, delimiter=",")
    rows = list(reader)
    for i in range(1, len(rows)):
      id_to_index[rows[i][1]] = int(rows[i][0])
    return id_to_index

# print(get_index_to_id())
# print(get_id_to_labels())

class AudiosetDataset(Dataset):
  def __init__(self, data_path, csv_path, num_classes=527, mixup_rate=1, mixup_alpha=0.5) -> None:
    # super().__init__()
    self.data_path = data_path
    self.mixup_rate = mixup_rate
    self.mixup_alpha = mixup_alpha
    self.num_classes = num_classes
    self.label_id_to_idx = get_id_to_index()
    self.file_id_to_idxs = {}
    self.file_id_to_idx = {}
    self.file_ids = []
    # for each entry in csv
    with open(csv_path) as csv_file:
      reader = csv.reader(csv_file, delimiter = " ")
      for row in reader:
        filename = get_audiofile_name_from_audioset_csv_row(row)
        if (os.path.exists(os.path.join(data_path, filename))):
          self.file_id_to_idx[filename] = len(self.file_ids)
          self.file_ids.append(filename)
          idxs = [self.label_id_to_idx[labelId] for labelId in row[3].strip("\"").split(",")]
          self.file_id_to_idxs[filename] = idxs
  def __getitem__(self, index):
    filename = self.file_ids[index]
    (waveform, _) = librosa.core.load(os.path.join(self.data_path, filename), sr = None)
    waveform = waveform[None, :]
    # print(waveform.size)
    result = np.zeros((1, 160_000))
    result[0, 0:min(waveform.shape[1], 160000)] = waveform[:waveform.shape[0], :min(waveform.shape[1], 160_000)]
    waveform = result
    # print(result.shape)
    if (random.random() < self.mixup_rate):
      other_idx = random.randint(0, self.__len__() - 1)
      other_filename = self.file_ids[other_idx]
      (other_waveform, _) = librosa.core.load(os.path.join(self.data_path, other_filename), sr = None)
      other_waveform = other_waveform[None, :]
      result = np.zeros((1, 160_000))
      result[0, 0:min(other_waveform.shape[1], 160000)] = other_waveform[:other_waveform.shape[0], :min(other_waveform.shape[1], 160000)]
      other_waveform = result
      lmbda = np.random.beta(self.mixup_alpha, self.mixup_alpha)
      waveform = self.mixup(waveform, other_waveform, lmbda)
      y = np.zeros(self.num_classes)
      for label_idx in self.file_id_to_idxs[filename]:
        y[label_idx] = lmbda
      for label_idx in self.file_id_to_idxs[other_filename]:
        y[label_idx] += (1 - lmbda)
      y /= np.max(y)
    else:
      y = np.zeros(self.num_classes)
      for label_idx in self.file_id_to_idxs[filename]:
        y[label_idx] = 1.0
    return waveform, y

  def mixup(self, waveform1, waveform2, lmbda):
    return lmbda * waveform1 + (1 - lmbda) * waveform2

  def __len__(self):
    return len(self.file_ids)