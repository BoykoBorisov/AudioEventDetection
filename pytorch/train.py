from ssl import OP_NO_RENEGOTIATION
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time
import os

from utils import get_stats

def knowledge_distilation_loss_fn(teacher_inference_temperature, teacher_inference_weight): 
  teacher_inference_temperature = teacher_inference_temperature
  teacher_inference_weight = teacher_inference_weight
  def loss_fn(student_inference, teacher_inference, ground_truth):
    teacher_inference = teacher_inference / teacher_inference_temperature
    teacher_inference = F.softmax(teacher_inference, dim=1)
    student_inference_distilation = F.softmax(student_inference / teacher_inference_temperature, dim=1)
    # Multiply by square of temperature to scale it up, this technique is mentioned in the original paper
    # https://arxiv.org/pdf/1503.02531.pdf
    soft_target_loss = F.kl_div(student_inference_distilation, teacher_inference, reduction="batchmean") * teacher_inference_temperature * teacher_inference_temperature
    ground_truth_loss = F.cross_entropy(student_inference, ground_truth)
    return teacher_inference_weight * soft_target_loss + (1 - teacher_inference_weight) * ground_truth_loss
  return loss_fn


def save_model(name, map, best_mAP, model, dir_path):
  file_name = "model_params_" + str(name) + ".pth"
  file_path = os.path.join(dir_path, file_name)
  state_dict = model.state_dict()
  torch.save(state_dict, file_path)
  if map > best_mAP:
    file_name = "model_params_" + str(name) + ".pth"
    file_path = os.path.join(dir_path, file_name)
    torch.save(state_dict, file_path)

def weight_average(model, dir_path, start_epoch, end_epoch, dataloader_validation, best_mAP):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  file_name = "model_params" + end_epoch + ".pth"
  file_path = os.path.join(dir_path, file_name)
  state_dict = torch.load(file_path)
  for epoch in range(start_epoch, end_epoch):
    file_name = "model_params" + epoch + ".pth"
    file_path = os.path.join(dir_path, file_name)
    other_epoch_state_dict =  torch.load(file_path)
    for key in state_dict:
      state_dict[key] += other_epoch_state_dict[key]
  
  lens = end_epoch - start_epoch + 1
  for key in state_dict:
    state_dict[key] /= lens
  
  model.load_state_dict(state_dict)
  # validation
  ground_truth_validation = []
  prediction_validation = []

  with torch.no_grad():
    for (batch_waveforms, ground_truth_labels) in dataloader_validation:
      batch_waveforms = batch_waveforms.to(device)
      batch_labels = batch_labels.to(device)
      y_hat = model(batch_waveforms)
      prediction_validation.append(y_hat.cpu().detach())
      ground_truth_validation.append(ground_truth_labels.cpu().detach())

      ground_truth_validation = torch.cat(ground_truth_validation)
      prediction_validation = torch.cat(prediction_validation)
      stats = get_stats(prediction_validation, ground_truth_validation)
      mAP = stats["MAP"]
  save_model("weight_averaging", mAP, best_mAP, model, dir_path)

def train(model, teacher_model, dataloader_training, dataloader_validation, epoch_count, 
          learning_rate, learning_rate_decay, learning_rate_dacay_step, warmup_iterations, 
          teacher_inference_weight, teacher_inference_temperature, should_apply_weight_averaging, 
          weight_averaging_start_epoch, weight_averaging_end_epoch, dir_path_save_model_weights
        ):
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('running on ' + str(device))
        
        best_mAP = 0
        best_epoch = 0

        model = model.to(device)
        teacher_model = teacher_model.to(device)
        optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay=5e-8)
        scheduler = optim.lr_scheduler.StepLR(optimizer, learning_rate_dacay_step, learning_rate_decay)
        iteration_count = 0
        loss_fn = knowledge_distilation_loss_fn(teacher_inference_temperature, teacher_inference_weight)
        print ("Starting training")
        for epoch in range(epoch_count):
          epoch_start_time = time.time()
          total_epoch_loss = 0
          # Tell the model you are training it, affects how the built in dropout layers of
          # EfficientNet behave
          model.train()

          for iteration, (batch_waveforms, batch_labels) in enumerate(dataloader_training):
            batch_waveforms = batch_waveforms.float()
            batch_waveforms = torch.squeeze(batch_waveforms)
            batch_waveforms = batch_waveforms.to(device)
            batch_labels = batch_labels.to(device)
            if (iteration_count < warmup_iterations):
              # Gradual warmup strategy
              current_learning_rate = (iteration_count / warmup_iterations) * learning_rate
              for group in optimizer.param_groups:
                group["lr"] = current_learning_rate
              print(f"Warm up lr {current_learning_rate}", flush=True)

            prediction_teacher = teacher_model(batch_waveforms)
            y_hat = model(batch_waveforms)
            loss = loss_fn(y_hat, prediction_teacher, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_epoch_loss += loss.item()
            if (iteration_count % 10 == 0):
              print(f"Epoch {epoch}: iteration {iteration}, loss this batch: {loss}", flush=True)
            iteration_count += 1
            break
          model.eval()
          # VALIDATION
          ground_truth_validation = []
          prediction_validation = []

          with torch.no_grad():
            for (batch_waveforms, ground_truth_labels) in dataloader_validation:
              batch_waveforms = batch_waveforms.float()
              batch_waveforms = torch.squeeze(batch_waveforms)
              batch_waveforms = batch_waveforms.to(device)
              ground_truth_labels = ground_truth_labels.to(device)
              y_hat = model(batch_waveforms)
              prediction_validation.append(y_hat.cpu().detach())
              ground_truth_validation.append(ground_truth_labels.cpu().detach())

            ground_truth_validation = torch.cat(ground_truth_validation)
            prediction_validation = torch.cat(prediction_validation)
            stats = get_stats(prediction_validation, ground_truth_validation)
            map = stats["MAP"]
          save_model(epoch, map, best_mAP, model, dir_path_save_model_weights)
          epoch_duration_minutes = (time.time() - epoch_start_time) / 60
          print(f"EPOCH: {epoch} | MaP: {map} | EPOCH DURATION {epoch_duration_minutes}")
        
        if should_apply_weight_averaging:
          weight_average(model, weight_averaging_start_epoch, weight_averaging_end_epoch, dataloader_validation, best_mAP)

          ground_truth_validation = []
          prediction_validation = []

          for (batch_waveforms, ground_truth_labels) in dataloader_validation:
            batch_waveforms = batch_waveforms.to(device)
            batch_labels = batch_labels.to(device)
            y_hat = model(batch_waveforms)
            prediction_validation.append(y_hat.cpu().detach())
            ground_truth_validation.append(ground_truth_labels.cpu().detach())

          ground_truth_validation = torch.cat(ground_truth_validation)
          prediction_validation = torch.cat(prediction_validation)
          stats = get_stats(prediction_validation, ground_truth_validation)
          map = stats["MAP"]
          save_model(epoch, map, best_mAP, model, dir_path_save_model_weights)

        

        