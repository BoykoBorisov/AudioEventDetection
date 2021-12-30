from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import get_same_padding_conv2d, Conv2dStaticSamePadding
from numpy import select
import librosa
from model import EfficientAudioNet
from torch.nn.functional import sigmoid
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from utils import move_data_to_device, get_stats
from audioset_weight_generator import generate_weights, get_sampler
from audioset_dataset import AudiosetDataset
from hear21passt.hear21passt.base import load_model, get_scene_embeddings, get_timestamp_embeddings
import torch
import numpy as np
import os
import wave

# generate_weights("/Users/boykoborisov/Desktop/Uni/ThirdYearProject/data-loader2/unbalanced_train_segments.csv",
#                  "/Users/boykoborisov/Desktop/Uni/ThirdYearProject/data-loader2/output",
#                        "weights.csv")


# dataset = AudiosetDataset("/Users/boykoborisov/Desktop/Uni/ThirdYearProject/data-loader2/output",
                # "/Users/boykoborisov/Desktop/Uni/ThirdYearProject/data-loader2/unbalanced_train_segments.csv",
              # )

# print(get_sampler(dataset, r"/Users/boykoborisov/Desktop/Uni/ThirdYearProject/weights.csv"))

# print(get_stats(np.random.rand(100, 100), np.random.randint(low = 0, high = 2, size = (100,100))))

root_dir = r"/Users/boykoborisov/Desktop/Uni/ThirdYearProject/data-loader2/output"
error_count = 0
full_count = 0
for file_name in os.listdir(root_dir):
  try:
    full_count += 1
    file = wave.open(os.path.join(root_dir, file_name))
    file.close()
  except:
    error_count += 1
    print(file_name, error_count, "/", full_count)
    os.remove(os.path.join(root_dir, file_name))
# print(count)
  