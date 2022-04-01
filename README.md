# Deep Learnig for Audio Event Detection
## A third year project by Boyko Borisov

## This project contains three deliverables:
1. A pipeline for defining and training an augmented version of EfficientNet for audio tagging with knowledge distillation
2. A tool for downloading audio files from the Audioset dataset
3. A GUI to visualise the probability distribution produced by a trained model via 1. on real-time recordings 

### 0. Install python packages - needed for all recepies 
  1. (Optional) start and activate a Python virtual environment
    
    python3 -m venv env
    source env/bin/activate 

  2. Install all necessary packages from requirements.txt
    
    pip install -r requirements.txt

### Recipe 1: Instructions for running the training procedure with Audioset

  1.1 Ensure you have .wav data samples from Audioset and the Audioset CSV (you can use deliverable 2 for that)

  1.2 Enter pytorch folder:

    cd pytorch

  1.3 Generate the sampler weights by running:

    python3 audioset_weight_generator --arguments

    Run python3 audioset_weight_generator --help to see the arguments that need to be provided

  1.3 Configure the exact training parameters in run_train.py, this includes specifying the following directory paths:
    
  * dir_path_save_model_weights - directory, where the model weights will be saved during training
  
  * dir_path_sample_weights - path to the CSV file containing the sampling weights for each training sample

  * dir_path_samples_training - path to the directory containing the .wav files that will be used for training

  * dir_path_sample_validation - path to the directory containing the .wav files that will be used for validation

  * csv_path_training_samples - the Audioset dataset of the training samples

  * csv_path_validation_samples - the Audioset dataset of the evaluation samples

  This can be done by either passing these as command-line arguments or by modifying the default values in run_train.

  Other hyperparameters that are going to be used in training can also be changed by providing them as command-line arguments. These include the hyperparameters for knowledge distillation, the mix-up rate and the learning rate among others. To see the exact way to provide these arguments run:

    python3 run_train.py --help

  1.4 (Optional) Set up a teacher network if you are going to train using knowledge distillation:

  This architecture is set up to run to fairly easily be integrated PaSST transformer as the teacher network. This can be done through running:

      pip install -e 'git+https://github.com/kkoutini/passt_hear21@0.0.8#egg=hear21passt' 
  
  Afterwards, uncomment lines 8 and 95 in run_train.py

### Recipe 2: Instructions for downloading audio files from the Audioset dataset
  2.1 Download and install youtube-dl and FFmpeg, instructions for that can be found at:

    http://ytdl-org.github.io/youtube-dl/download.html

    https://ffmpeg.org/download.html

  2.2 Download the Audioset dataset CSV for whichever segment of the dataset you would like to download. You can get the Audioset CSVs from:
    
    https://research.google.com/audioset/download.html
  
  2.3 Go to the data_loader subdirectory and change the Audioset path variable to match the path to where you downloaded the CSV from, part2. If you'd like to, you can change the directory destination for downloading the audio files.

  2.4 Run the loader.ipynb notebook


### Recipe 3: Instructions for starting the GUI
The application for data visualisation is in the form of a server-client web application. You will need to set up Node.js for the client. You will also need to have a weight file for the model, a weight file is currently provided at model_weights/best_weights. Currently, the app is configured to load these weights for the model without the need for adjustments.

  3.1 Install Node.js. and npm. For that go to:

    https://nodejs.org/en/download/

  3.2 Install all dependencies for the client:

    From a command line, enter the directory of the frontend and install the packages

        cd app/frontend
        npm install

  3.3 Go to the server directory

      cd ../backend
  
  3.4 In app.py, change the model weight path to the weights that will be loaded on the server to match the paths on your device

  3.5 Start the server

      python3 app.py
  
  3.6 In another command line, start the client

      cd app/frontend
      npm start

The screenshot folder contains a screenshot of the GUI running.

The code for this project was written from scratch. It has been, however, guided by several tutorials, frameworks' documentations, and repositories and therefore shares similarities. Resources include:
 * Aakash N S' PyTorch tutorials series - https://jovian.ai/aakashns/collections/deep-learning-with-pytorch
 * Chathuranga Siriwardhana's tutorial for using librosa - https://towardsdatascience.com/sound-event-classification-using-machine-learning-8768092beafc
 * Quinquang Kong's repository for PANNs for integrating mix-up - https://github.com/qiuqiangkong/audioset_classification
 * Yuan Gong's AST repository - https://github.com/YuanGongND/ast
 * Flask's documentation and tutorial for building http-servers - https://flask.palletsprojects.com/en/2.0.x/tutorial/
 * Aoife McDonagh's repositories related to acoustic scene recognition - https://github.com/aoifemcdonagh
 * Alexander Staisiuk's tutorial on weight averaging - https://stasiuk.medium.com/pytorch-weights-averaging-e2c0fa611a0c
 * PyTorch's extensive documentation, which includes many mini-tutorials - https://pytorch.org/docs/stable/index.html
