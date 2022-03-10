import argparse
from ast import arg


from audioset_dataset import get_id_to_index
from utils import get_audiofile_name_from_audioset_csv_row
import numpy as np
import csv
import os
import torch.utils.data as data

def generate_weights(csv_filepath, audiofiles_filepath, output_filepath, labels_num=527):
  label_id_to_idx = get_id_to_index()
  label_counts = np.zeros(labels_num)
  # Get number of samples per label
  with open(csv_filepath) as csv_file:
    reader = csv.reader(csv_file, delimiter = " ")
    for row in reader:
      filename = get_audiofile_name_from_audioset_csv_row(row)
      # print(filename)
      if (os.path.exists(os.path.join(audiofiles_filepath, filename))):
        labels = row[3].strip("\"").split(",")
        # print(labels)
        for label in labels:
          # print(type(label_id_to_idx[label]))
          label_counts[label_id_to_idx[label]] += 1
    # print(label_counts)
    calculate_weight = lambda f: 1000 / (f + 1)
    label_weights = calculate_weight(label_counts)
    # print(label_weights)
    # For each sample the weight is the sum of for each label in sample (10000 / samples per label
  with open(csv_filepath) as csv_file:
    reader = csv.reader(csv_file, delimiter = " ")
    sample_weights = []
    with open(output_filepath, "w") as output_file:
      writer = csv.writer(output_file, delimiter= " ")
      reader = csv.reader(csv_file, delimiter = " ")
      for row in reader:
        filename = get_audiofile_name_from_audioset_csv_row(row)
        if (os.path.exists(os.path.join(audiofiles_filepath,filename))):
          # print(filename)
          score = 0
          labels = row[3].strip("\"").split(",")
          for label in labels:
            score += label_weights[label_id_to_idx[label]]
          writer.writerow([filename, score])
          sample_weights.append(score)
  return sample_weights

def get_sampler(dataset, weights_filepath):
  weights = np.zeros(dataset.__len__())
  with open(weights_filepath) as csv_file:
    reader = csv.reader(csv_file, delimiter = " ")
    for row in reader:
      if row[0] in dataset.file_id_to_idx:
        weights[dataset.file_id_to_idx[row[0]]] = float(row[1])
    # print(weights[0:100])
  return data.WeightedRandomSampler(weights=weights, num_samples=dataset.__len__())

if __name__== '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--csv_filepath", type=str, help="Audioset filepath")
    parser.add_argument("--audiofiles_filepath", type=str, help="Path to the directory where the audiofiles are stored")
    parser.add_argument("--output_filepath", type=str, help="Path to where the csv with the sampler weights will be stored")
    args = parser.parse_args()
    generate_weights(args.csv_filepath, args.audiofiles_filepath, args.output_filepath)