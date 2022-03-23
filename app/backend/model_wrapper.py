from urllib import response
import torch

class Wrapper:
  def __init__(self, model, weights_path, index_to_label, index_to_id) -> None:
      model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
      for name, param in model.named_parameters():
        param.requires_grad = False
      self.model = model
      self.model.eval()
      self.index_to_label = index_to_label
      self.index_to_id = index_to_id

  def infer(self, waveform):
    waveform = torch.tensor(waveform)
    # waveform = torch.unsqueeze(waveform, 0)
    print(waveform.shape)
    inferences = self.model(waveform)
    inferences = inferences.detach().numpy()
    response = {}
    for (i, inference) in enumerate(inferences[0]):
      response[self.index_to_label[i]] = {"probabilities": inference.item(),
                                          "id": self.index_to_id[i]}
    return response
    

