# Audio tagging using deep neural networks
## A third year project by Boyko Borisov

## This project contains three deliverables:
1. A pipeline for defining and training an augmented version of EfficientNet for audio tagging with knowledge distilation
2. A tool for downloading audio files from the Audioset dataset
3. A GUI to visualise the probability distribution produced by the model on real time recordings 

### 0. Install python packages - needed for a all recepies 
  1. (Optional) start and activate a python virtual environment
    
    python3 -m venv env
    source env/bin/activate 

  2. Install all necessary packages from requirements.txt
    
    pip install -r requirements.txt
### 1. Instructions for running the training procedure with Audioset

  1.1 Ensure you have wav datasamples from Audioset and the Audioset csv (you can use deliverable 2 for that)

  1.2 Enter pytorch folder:

    cd pytorch

  1.3 Generate the sampler weights by running:

    python3 audioset_weight_generator --arguments

    Run python3 audioset_weight_generator --help to see the arguments that need to be provided

  1.3 Configure the exact training parameters in run_train.py, this includes specifying the following directory paths:
    
  * dir_path_save_model_weights - directory, where the model weights will be safed during training
  
  * dir_path_sample_weights - path to the csv file containing the sampling weights for each training sample

  * dir_path_samples_training - path to the directory containing the wav files that will be used for training

  * dir_path_sample_validation - path to the directory containing the wav files that will be used for validation

  * csv_path_training_samples - the Audioset dataset of the training samples

  * csv_path_validation_samples - the Audioset dataset of the evaluation samples

  This can be done by either passing these as arguments or by modifying the default values in run_train.

  1.4 (Optional) Set up teacher network if you are going to train using knowledge distilation:

  This architecture is set up to run with the PaSST transformer as the teacher network. This can be done through:

      pip install -e 'git+https://github.com/kkoutini/passt_hear21@0.0.8#egg=hear21passt' 

  1.5 Adjust the rest of the hyperparamenters in run_train according to the experiment you would like to run and run:

    python3 run_train.py

### 2. Instruction for downloading audio files from the Audioset dataset
  2.1 Download and install youtube-dl and ffmpeg, instructions for that can be found at:

    http://ytdl-org.github.io/youtube-dl/download.html

    https://ffmpeg.org/download.html

  2.2 Download the Audioset dataset csv for whichever segment of the dataset you would like to download. You can get the Audioset csvs from:
    
    https://research.google.com/audioset/download.html
  
  2.3 Go to the data_loader subdirectory and change the Audioset path variable to match the path to where you downloaded the csv from, part2. If you'd like to, you can change the directory destination for downloading the audiofiles.

  2.4 Run the loader.ipynb notebook


### 3. Instructions for starting the GUI
The application for data visualisation is in the form of a server-client web application. You will need to set up Node.js for the client and the flask python framework for the server. You will also need to have a weight file for the model.

  3.1 Install Node.js. and npm. For that go to:

    https://nodejs.org/en/download/

  3.2 Install all dependencies for the client:

    From a terminal, enter the directory of the frontend and install the packages

        cd app/frontend
        npm install

  3.3 Go to the server directory

      cd ../backend
  
  3.4 In app.py, change the model weight path to the weights that will be loaded on the server to match the paths on your device

  3.5 Start the server

      python3 app.py
  
  3.6 In another terminal, start the client

      cd app/frontend
      npm start
