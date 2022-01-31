# from model import EfficientAudioNet
# from time import time
# from efficientnet_pytorch import EfficientNet
# from efficientnet_pytorch.utils import get_same_padding_conv2d, Conv2dStaticSamePadding
# from numpy import select
# import librosa
# from scipy.sparse import data
# from torch.nn.functional import sigmoid
# from torchlibrosa.stft import Spectrogram, LogmelFilterBank
# from torchlibrosa.augmentation import SpecAugmentation
# from utils import move_data_to_device, get_stats
from audioset_weight_generator import generate_weights, get_sampler
# from audioset_dataset import AudiosetDataset
# from hear21passt.hear21passt.base import load_model, get_scene_embeddings, get_timestamp_embeddings
# from torch.utils.data import DataLoader
# import torch
# import numpy as np
# import os
# import wave

generate_weights("/home/jupyter/ThirdYearProject/datasets/Audioset/unbalanced_train_segments.csv",
                 "/home/jupyter/ThirdYearProject/data-loader2/output",
                 "/home/jupyter/ThirdYearProject/datasets/weights/weights_unbalanced.csv"
                )


# dataset = AudiosetDataset("/Users/boykoborisov/Desktop/Uni/ThirdYearProject/data-loader2/output",
#                 "/Users/boykoborisov/Desktop/Uni/ThirdYearProject/data-loader2/unbalanced_train_segments.csv",
#               )

# dataloader_training = DataLoader(dataset=dataset, batch_size=32, 
#                                    shuffle=False)
# student_model = EfficientAudioNet()
# teacher_model = load_model()

# all_mins = 0
# for i, (batch_waveforms, batch_labels) in enumerate(dataloader_training):
#   time_start = time()
#   batch_waveforms = batch_waveforms.float()
#   batch_waveforms = torch.squeeze(batch_waveforms)
#   print(teacher_model(batch_waveforms).size())
#   print(student_model(batch_waveforms).size())
#   print(batch_labels.size())
#   batch_time = time() - time_start
#   print(batch_time)
#   all_mins += batch_time
# print(all_mins)

# waveform = torch.tensor(dataset.__getitem__(6)[0], dtype=torch.float)
# print(model(waveform))
# print(get_sampler(dataset, r"/Users/boykoborisov/Desktop/Uni/ThirdYearProject/weights.csv"))

# print(get_stats(np.random.rand(100, 100), np.random.randint(low = 0, high = 2, size = (100,100))))

# root_dir = r"/home/jupyter/ThirdYearProject/data-loader2/output_eval"
# error_count = 0
# full_count = 0
# for file_name in os.listdir(root_dir):
#   try:
#     full_count += 1
#     file = wave.open(os.path.join(root_dir, file_name))
#     file.close()
#   except:
#     error_count += 1
#     print(file_name, error_count, "/", full_count)
#     os.remove(os.path.join(root_dir, file_name))
# print(error_count)
  