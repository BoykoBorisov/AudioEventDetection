from flask import Flask, request
from flask_cors import CORS
import librosa
from model_wrapper import Wrapper
import sys
import numpy as np

sys.path.insert(0,'../../pytorch')
from model import EfficientAudioNet
from audioset_dataset import get_index_to_label

app = Flask(__name__)
CORS(app)

model = EfficientAudioNet() 
weights_path = r"/Users/boykoborisov/Desktop/Uni/ThirdYearProject/model_weights/best_weights/model_params_90021158_0.41221821796293234.pth"
index_to_label = get_index_to_label()
wrapper = Wrapper(model, weights_path, index_to_label)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/test")
def test():
  return "test"

@app.route("/infer", methods=["POST"])
def infer():
  # (waveform1, _) = librosa.core.load("/Users/boykoborisov/Desktop/Uni/ThirdYearProject/data-loader2/output/---1_cCGK4M_0.000.wav", sr=None)
  # return {"aaa": 55}
  if "1" in request.files:
    # print("Found 1.wav")
    # print(request.files["1"])
    request.files["1"].save("/Users/boykoborisov/Desktop/Uni/ThirdYearProject/app/backend/1.wav")
  # print(waveform1.shape)
  (waveform2, _) = librosa.load("/Users/boykoborisov/Desktop/Uni/ThirdYearProject/app/backend/1.wav", sr = 16_000)
  print(waveform2.shape)
  waveform2 = waveform2[None, :]
  print(waveform2.shape)
  # waveform2.shape
  result = np.zeros((1, 160_000))
  result[0, 0:min(waveform2.shape[1], 160000)] = waveform2[:waveform2.shape[0], :min(waveform2.shape[1], 160_000)]
  print(result.shape)
  waveform2 = result[0]
  print(waveform2.shape)
  res = wrapper.infer(waveform2)
  # print(res)
  return res

if __name__ == '__main__':
   app.run()