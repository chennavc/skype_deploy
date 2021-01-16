from flask import Flask,render_template,request
import sounddevice as sd
from scipy.io.wavfile import write
import soundfile as sf

import librosa
import librosa.display

import numpy as np
import cv2
import os
import keras
from keras import layers, models, optimizers
from keras import backend as K
from IPython.display import display, Image

import time
from time import sleep

import soundfile
import librosa

import librosa.display

import matplotlib.pyplot as plt

import cv2
# from flask_jsonpify import jsonpify

app = Flask(__name__)

def rec():
    fs = 44100  # Sample rate
    seconds =60# Duration of recording
    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    
    sd.wait()  # Wait until recording is finished
    write('test.wav', fs, myrecording)  # Save as WAV file 
    return('test.wav')

def mel_spec(path):
    y, sr = librosa.load(path)
    librosa.feature.melspectrogram(y=y, sr=sr)
    D = np.abs(librosa.stft(y))**2
    S = librosa.feature.melspectrogram(S=D)
    # Passing through arguments to the Mel filters
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                        fmax=8000)
    plt.figure(figsize=(10,8))
    librosa.display.specshow(librosa.power_to_db(S,
                                                 ref=np.max),
                             fmax=8000,
                              )
    plt.savefig("test.png")
    return("test.png")
def array (impath):
    img=cv2.imread(impath)
    img=cv2.resize(img,(200,200))
    return img
def model(arr):
    reconstructed_model = keras.models.load_model("amodel")
    result=reconstructed_model.predict_classes(arr.reshape(-1, 200, 200,3),verbose=2)
    if result == 1:
        return("genuine")
    else:
        return("fraud")

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/result",methods=['GET','POST'])
def result():
    if request.method == 'POST':
        # symbol=request.form['symbol']

        
        result=model(array (mel_spec(rec())))
        

        return render_template ('result.html',result=result)
    else:
        return render_template('index.html')
@app.route("/comingsoon",methods=['GET','POST'])
def comingsoon ():
    return render_template('comingsoon.html')



if __name__ == "__main__":
    app.run(debug=True)