from flask import Flask, request
from flask_cors import CORS
import librosa
from model_wrapper import Wrapper
import sys
import numpy as np

sys.path.insert(0,'../../pytorch')
from model import EfficientAudioNet
from audioset_dataset import get_index_to_label, get_index_to_id
import os

app = Flask(__name__)
CORS(app, )

model = EfficientAudioNet() 

weights_path = os.path.realpath(__file__)[:-6] + "../../model_weights/best_weights/best_weight_with_KD_no_WA.pth"
index_to_label = get_index_to_label()
index_to_id = get_index_to_id()
wrapper = Wrapper(model, weights_path, index_to_label, index_to_id)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/test")
def test():
  return "test"

@app.route("/infer", methods=["POST"])
def infer():
  if "1" in request.files:
    request.files["1"].save("1.wav")

  (waveform, _) = librosa.core.load("1.wav", sr = None)
  waveform = waveform[None, :]
  result = np.zeros((1, 160_000), dtype=np.float32)
  result[0, 0:min(waveform.shape[1], 160_000)] = waveform[:waveform.shape[0], :min(waveform.shape[1], 160_000)]
  waveform = result
  res = wrapper.infer(waveform)
  return res

@app.after_request
def after_request(response):
    header = response.headers
    header['Access-Control-Allow-Origin'] = '*'
    header['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    header['Access-Control-Allow-Methods'] = 'OPTIONS, HEAD, GET, POST, DELETE, PUT'
    return response
    
if __name__ == '__main__':
   app.run()
