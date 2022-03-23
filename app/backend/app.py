from flask import Flask, request
from flask_cors import CORS
import librosa
from model_wrapper import Wrapper
import sys
import numpy as np

sys.path.insert(0,'../../pytorch')
from model import EfficientAudioNet
from audioset_dataset import get_index_to_label, get_index_to_id

app = Flask(__name__)
CORS(app, )

model = EfficientAudioNet() 
weights_path = r"/Users/boykoborisov/Desktop/Uni/ThirdYearProject/model_weights/best_weights/model_params_90021158_0.41221821796293234.pth"
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
  # (waveform1, _) = librosa.core.load("/Users/boykoborisov/Desktop/Uni/ThirdYearProject/data-loader2/output/---1_cCGK4M_0.000.wav", sr=None)
  # return {"aaa": 55}
  if "1" in request.files:
    # print(request.files["1"])
    request.files["1"].save("/Users/boykoborisov/Desktop/Uni/ThirdYearProject/app/backend/1.wav")

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
