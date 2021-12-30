from audioset_dataset import AudiosetDataset
from torch.utils.data import DataLoader
from audioset_weight_generator import get_sampler
from model import EfficientAudioNet
from hear21passt.hear21passt.base import load_model
if __name__== '__main__':
  # hyperparameters for training
  epoch_count = 30
  learning_rate = 0
  learning_rate_decay = 0
  learning_rate_dacay_step = 0
  batch_size = 128 

  # hyperparameters for knowledge distilation
  teacher_inference_weight = 0
  teacher_inference_temperature = 0

  # hyperparameters for mixup
  mixup_rate = 0
  mixup_weight = 0

  # hyperparameters for weight averaging 
  should_apply_weight_averaging = True
  weight_averaging_start_epoch = 20
  weight_averaging_end_epoch = 30

  num_classes = 527
  efficientnet_size = 2

  dir_path_save_model_weights = r"/Users/boykoborisov/Desktop/Uni/ThirdYearProject/model_weights"
  dir_path_sample_weights = r"/Users/boykoborisov/Desktop/Uni/ThirdYearProject/weights.csv"
  dir_path_samples_training = r"/Users/boykoborisov/Desktop/Uni/ThirdYearProject/data-loader2/output"
  dir_path_sample_validation = r"/Users/boykoborisov/Desktop/Uni/ThirdYearProject/data-loader2/output_eval"

  csv_path_training_samples = r"/Users/boykoborisov/Desktop/Uni/ThirdYearProject/datasets/Audioset/unbalanced_train_segments.csv"
  csv_path_validation_samples = r"/Users/boykoborisov/Desktop/Uni/ThirdYearProject/datasets/Audioset/eval_segments.csv"

  dataset_training = AudiosetDataset(data_path=dir_path_samples_training, csv_path=csv_path_training_samples, 
                                     num_classes=num_classes, mixup_rate=mixup_rate, mixup_alpha=mixup_weight)
  dataset_validation = AudiosetDataset(data_path=dir_path_sample_validation, csv_path=csv_path_validation_samples,
                                       num_classes=num_classes, mixup_rate=0, mixup_alpha=0) 
  weighted_sampler_training = get_sampler(dataset_training, dir_path_sample_weights)


  dataloader_training = DataLoader(dataset=dataset_training, batch_size=batch_size, 
                                   shuffle=False, sampler=weighted_sampler_training)

  dataloader_validation = DataLoader(dataset=dataset_validation, batch_size = batch_size, 
                                    shuffle=False)

  model = EfficientAudioNet()
  teacher_model = load_model()

  for i, (data, label) in enumerate(dataloader_training):
    print(i, data.shape)


  