from urllib import response
import torch

class Wrapper:
  def __init__(self, model, weights_path, index_to_label) -> None:
      model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
      for name, param in model.named_parameters():
        param.requires_grad = False
      self.model = model
      self.index_to_label = index_to_label

  def infer(self, waveform):
    waveform = torch.tensor(waveform, dtype=torch.float)
    waveform = torch.unsqueeze(waveform, 0)
    inferences = self.model(waveform)
    inferences = inferences.detach().numpy()
    response = {}
    for (i, inference) in enumerate(inferences[0]):
      response[self.index_to_label[i]] = {"probabilities": inference.item(),
                                          "id": i}
    return response
    

