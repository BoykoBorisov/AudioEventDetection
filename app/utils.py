import csv
import json
import os
from math import ceil, log, log10
from re import I


if __name__ == '__main__':
  id_to_label = {}
  index_to_id = {}
  with open(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets/Audioset/class_labels_indices.csv"))) as file:
    reader = csv.reader(file, delimiter=",")
    rows = list(reader)
    for i in range(1, len(rows)):
      id_to_label[rows[i][1]] = rows[i][2]
      index_to_id[rows[i][0]] = rows[i][1]

  classes = {}

  with open("/Users/boykoborisov/Desktop/Uni/ThirdYearProject/pytorch/temp.txt") as dataset_csv:
    reader = csv.reader(dataset_csv, delimiter= " ")
    for row in reader:
      classes[index_to_id[row[0]]] = row[1]
 
    # print(f"{k} :  {classes[k]}, {log(classes[k])}")
  f = open("coefs.json", "w")
  json.dump(classes, fp=f, indent=2)
  f.close()

  