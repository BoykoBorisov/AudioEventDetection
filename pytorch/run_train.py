from tkinter import E
from audioset_dataset import AudiosetDataset
from torch.utils.data import DataLoader
from audioset_weight_generator import get_sampler
from model import EfficientAudioNet
from hear21passt.hear21passt.base import load_model
from train import train
if __name__== '__main__':
  # hyperparameters for training
  epoch_count = 26
  learning_rate = 0.001
  learning_rate_decay = 1
  learning_rate_dacay_step = 14
  batch_size = 40
  warmup_iterations = 100

  # hyperparameters for knowledge distilation
  teacher_inference_weight = 0.4
  teacher_inference_temperature = 2

  # hyperparameters for mixup
  mixup_rate = 0
  mixup_weight = 0.5

  # hyperparameters for weight averaging 
  should_apply_weight_averaging = True
  weight_averaging_start_epoch = 20
  weight_averaging_end_epoch = 25

  num_classes = 527
  efficientnet_size = 2

  dir_path_save_model_weights = r"/home/jupyter/ThirdYearProject/model_weights"
  dir_path_sample_weights = r"/home/jupyter/ThirdYearProject/datasets/weights/weights_bal_train.csv"
  dir_path_samples_training = r"/home/jupyter/ThirdYearProject/data-loader2/output_bal_train"
  dir_path_sample_validation = r"/home/jupyter/ThirdYearProject/data-loader2/output_eval"

  csv_path_training_samples = r"/home/jupyter/ThirdYearProject/datasets/Audioset/balanced_train_segments.csv"
  csv_path_validation_samples = r"/home/jupyter/ThirdYearProject/datasets/Audioset/eval_segments.csv"

  dataset_training = AudiosetDataset(data_path=dir_path_samples_training, csv_path=csv_path_training_samples,
                                     num_classes=num_classes, mixup_rate=mixup_rate, mixup_alpha=mixup_weight)

  dataset_validation = AudiosetDataset(data_path=dir_path_sample_validation, csv_path=csv_path_validation_samples,
                                       num_classes=num_classes, mixup_rate=0, mixup_alpha=0)

  weighted_sampler_training = get_sampler(dataset_training, dir_path_sample_weights)

  #sampler option acts like shuffle
  dataloader_training = DataLoader(dataset=dataset_training, batch_size=batch_size, 
                                   shuffle=False, 
                                   sampler=weighted_sampler_training,
                                   pin_memory=False, num_workers=4)

  dataloader_validation = DataLoader(dataset=dataset_validation, batch_size = 45, 
                                    shuffle=False, pin_memory=True, num_workers=4)

  model = EfficientAudioNet()
  
  #hear_passt_model
  # teacher_model = load_model()

  # for name, param in teacher_model.named_parameters():
    # param.requires_grad = False
  teacher_model = None

  train(model=model, teacher_model=teacher_model, dataloader_training=dataloader_training,
        dataloader_validation=dataloader_validation, epoch_count=epoch_count, learning_rate=learning_rate,
        learning_rate_decay=learning_rate_decay, learning_rate_dacay_step=learning_rate_dacay_step, warmup_iterations=warmup_iterations,
        teacher_inference_weight=teacher_inference_weight, teacher_inference_temperature=teacher_inference_temperature,
        should_apply_weight_averaging=should_apply_weight_averaging, weight_averaging_start_epoch=weight_averaging_start_epoch, 
        weight_averaging_end_epoch=weight_averaging_end_epoch, dir_path_save_model_weights=dir_path_save_model_weights,
        resume_training=False, resume_training_weights_path = "/home/jupyter/ThirdYearProject/model_weights/model_params_6_0.16051092838844316.pth", 
        resume_epoch = 7)
  