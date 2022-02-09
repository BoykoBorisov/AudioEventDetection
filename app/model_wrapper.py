from urllib import response
import torch

class Wrapper:
  def __init__(self, model, weights_path, index_to_label) -> None:
      model.load_state_dict(torch.load(weights_path))
      self.model = model
      self.index_to_label = index_to_label

  def infer(self, waveform):
    inferences = self.model(waveform)
    inferences = inferences.detach().numpy()
    response = []
    for inference in inferences:
      response.append(dict(zip(self.index_to_label, inference)))
    return response
    

