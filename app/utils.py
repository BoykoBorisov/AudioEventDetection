import csv
import json
from math import ceil, log, log10


if __name__ == '__main__':
  file_set = set()
  classes = {}
  with open("/Users/boykoborisov/Desktop/Uni/ThirdYearProject/downloads.txt") as filenames:
    for filename in filenames:
      file_set.add(filename[:11])

  with open("/Users/boykoborisov/Desktop/Uni/ThirdYearProject/datasets/Audioset/unbalanced_train_segments.csv") as dataset_csv:
    reader = csv.reader(dataset_csv, delimiter= " ")
    for row in reader:
        if row[0][:-1] in file_set:
          sample_classes = row[3].split(",")
          for sample_class in sample_classes:
            if sample_class in classes:
              classes[sample_class] += 1
            else:
              classes[sample_class] = 1 

    # print(len(classes))
  # for (k,v) in sorted(classes.items(), key=lambda item: item[1]):
  #   print(f"{k} : {v}")
  coeficients = {k: ceil(((log(v, 1500)) ** 4) / 3) for (k, v) in classes.items()}
    # print(f"{k} :  {classes[k]}, {log(classes[k])}")
  f = open("coefs.json", "w")
  json.dump(coeficients, fp=f, indent=2)
  f.close()

  