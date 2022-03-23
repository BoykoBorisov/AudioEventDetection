import librosa
from scipy.signal.spectral import spectrogram
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation
from time import time

from utils import move_data_to_device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
default_params = dict(
  sample_rate = 16000,
  window_size = 1024,
  hop_size = 160,
  mel_bins = 128,
  fmin = 50,
  fmax = 14000,
  num_classes=527
)
class EfficientAudioNet(nn.Module):
  def __init__(self, sound_params=None, pretrained_efficient_net = True, efficient_net_type = "efficientnet-b2"):
    super().__init__()
    if sound_params == None:
      self.sample_rate = default_params["sample_rate"]
      self.window_size = default_params["window_size"]
      self.hop_size = default_params["hop_size"]
      self.mel_bins = default_params["mel_bins"]
      self.fmin = default_params["fmin"]
      self.fmax = default_params["fmax"]
      self.num_classes = default_params["num_classes"]
    else:
      self.sample_rate = sound_params["sample_rate"]
      self.window_size = sound_params["window_size"]
      self.hop_size = sound_params["hop_size"]
      self.mel_bins = sound_params["mel_bins"]
      self.fmin = sound_params["fmin"]
      self.fmax = sound_params["fmax"]
      self.num_classes = sound_params["num_classes"]
      
    self.spectrogram_extractor = Spectrogram(n_fft=self.window_size, 
                                             hop_length=self.hop_size, 
                                             win_length=self.window_size)

    self.logmel_extractor = LogmelFilterBank(sr=self.sample_rate, 
                                             n_fft=self.window_size, 
                                             n_mels=self.mel_bins, 
                                             fmin=self.fmin, 
                                             fmax=self.fmax,
                                             top_db=None)


    self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
          freq_drop_width=8, freq_stripes_num=2)

    spectrogram_size = (int(self.sample_rate / self.hop_size), self.mel_bins)

    if (pretrained_efficient_net):
      self.efficient_net = EfficientNet.from_pretrained(efficient_net_type, in_channels=1, image_size = spectrogram_size, num_classes=self.num_classes)
    else:
      self.efficient_net = EfficientNet.from_name(efficient_net_type, in_channels=1, image_size = spectrogram_size, num_classes=self.num_classes)

    self.sigmoid = nn.Sigmoid()

  def forward(self, x):
    x = self.spectrogram_extractor(x)
    x = self.logmel_extractor(x)
    # print(self.training)
    if (self.training):
      x = self.spec_augmenter(x)
    x = self.efficient_net(x)
    x = self.sigmoid(x)
    return x


if __name__ == "__main__":
  (waveform, _) = librosa.core.load("/Users/boykoborisov/Desktop/Uni/ThirdYearProject/data-loader2/output/--_CyXnrDW0_30.000.wav", sr=None)
  waveform = waveform[None, :]
  waveform = move_data_to_device(waveform, device)
  print(waveform.size())
  model = EfficientAudioNet()
  with torch.no_grad():
    t = time()
    print(model(waveform).size())
    print((time() - t))