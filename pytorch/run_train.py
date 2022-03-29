import argparse

import torch
from audioset_dataset import AudiosetDataset
from torch.utils.data import DataLoader
from audioset_weight_generator import get_sampler
from model import EfficientAudioNet
# from hear21passt.hear21passt.base import load_model
# from panns_inference import AudioTagging, models
from train import train, weight_average_selected_states

if __name__== '__main__':
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)


  parser.add_argument("--epoch_count", type=int, default=10, help="Number of epochs spent training")
  parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")
  parser.add_argument("--learning_rate_decay", type=float, default=0.5, help="The number the learning rate will be multiplied by every x epochs")
  parser.add_argument("--learning_rate_decay_step", type=int, default=3, help="Number of epochs between the learning rate decays")
  parser.add_argument("--batch_size", type=int, default=28)
  parser.add_argument("--warmup_iterations", type=int, default=1000, help="Number of iterations when warm up will be applied")
  
  parser.add_argument("--teacher_inference_weight", type=float, default=0.1, help="Teacher inference weight")
  parser.add_argument("--teacher_inference_temperature", type=int, default=2, help="Teacher inference weight")
  
  parser.add_argument("--mixup_rate", type=float, default=0.7, help="Mixup rate, what percentage of the samples will be a subject to mixup")
  parser.add_argument("--mixup_weight", type=float, default=0.5, help="How much should the mixup affect the sample?")

  parser.add_argument("--should_apply_weight_averaging", type=bool, default=False)
  parser.add_argument("--weight_averaging_start", type=int, default= 0)
  parser.add_argument("--weight_averaging_end", type=int, default= 0)
  
  parser.add_argument("--dir_path_save_model_weights", type=str, default=r"/home/jupyter/ThirdYearProject/model_weights")
  parser.add_argument("--dir_path_sample_weights", type=str, default=r"/home/jupyter/ThirdYearProject/weights.csv")
  parser.add_argument("--dir_path_samples_training", type=str, default=r"/home/jupyter/ThirdYearProject/data_loader/output")
  parser.add_argument("--dir_path_sample_validation", type=str, default=r"/home/jupyter/ThirdYearProject/data_loader/output_eval")
  parser.add_argument("--csv_path_training_samples", type=str, default=r"/home/jupyter/ThirdYearProject/datasets/Audioset/unbalanced_train_segments.csv")
  parser.add_argument("--csv_path_validation_samples", type=str, default=r"/home/jupyter/ThirdYearProject/datasets/Audioset/eval_segments.csv")

  args = parser.parse_args()

  print(args.epoch_count)
  # hyperparameters for training
  epoch_count = args.epoch_count
  learning_rate = args.learning_rate
  learning_rate_decay = args.learning_rate_decay
  learning_rate_dacay_step = args.learning_rate_decay_step
  batch_size = args.batch_size
  warmup_iterations = args.warmup_iterations

  # hyperparameters for knowledge distilation
  teacher_inference_weight = args.teacher_inference_weight
  teacher_inference_temperature = args.teacher_inference_temperature

  # hyperparameters for mixup
  mixup_rate = args.mixup_rate
  mixup_weight = args.mixup_weight

  # hyperparameters for weight averaging 
  should_apply_weight_averaging = args.should_apply_weight_averaging
  weight_averaging_start_epoch = args.weight_averaging_start
  weight_averaging_end_epoch = args.weight_averaging_end

  num_classes = 527
  efficientnet_size = 2

  dir_path_save_model_weights = args.dir_path_save_model_weights
  dir_path_sample_weights = args.dir_path_sample_weights
  dir_path_samples_training = args.dir_path_samples_training
  dir_path_sample_validation = args.dir_path_sample_validation
  csv_path_training_samples = args.csv_path_training_samples
  csv_path_validation_samples = args.csv_path_validation_samples

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

  dataloader_validation = DataLoader(dataset=dataset_validation, batch_size = 40, 
                                    shuffle=False, pin_memory=True, num_workers=4)

  model = EfficientAudioNet()
      
  #hear_passt_model
  teacher_model = None


  train(model=model, teacher_model=teacher_model, dataloader_training=dataloader_training,
        dataloader_validation=dataloader_validation, epoch_count=epoch_count, learning_rate=learning_rate,
        learning_rate_decay=learning_rate_decay, learning_rate_dacay_step=learning_rate_dacay_step, warmup_iterations=warmup_iterations,
        teacher_inference_weight=teacher_inference_weight, teacher_inference_temperature=teacher_inference_temperature,
        should_apply_weight_averaging=should_apply_weight_averaging, weight_averaging_start_epoch=weight_averaging_start_epoch, 
        weight_averaging_end_epoch=weight_averaging_end_epoch, dir_path_save_model_weights=dir_path_save_model_weights, stop_knowledge_distilation = None,
        resume_training=True, resume_training_weights_path = "ThirdYearProject/model_weights/best_map_model_params_5.pth", 
        resume_epoch = 6)
  