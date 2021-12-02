from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import get_same_padding_conv2d, Conv2dStaticSamePadding
from numpy import select
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from utils import move_data_to_device
from audioset_weight_generator import generate_weights
from audioset_dataset import AudiosetDataset
# print(generate_weights("/Users/boykoborisov/Desktop/Uni/ThirdYearProject/data-loader2/unbalanced_train_segments.csv",
#                  "/Users/boykoborisov/Desktop/Uni/ThirdYearProject/data-loader2/output",
#                  "output.csv"))


dataset = AudiosetDataset("/Users/boykoborisov/Desktop/Uni/ThirdYearProject/data-loader2/output",
                "/Users/boykoborisov/Desktop/Uni/ThirdYearProject/data-loader2/unbalanced_train_segments.csv")

# print(dataset.__getitem__(100))