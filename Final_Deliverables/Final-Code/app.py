""" Building the user interface application that enables the user to interact.
In this application the person is required to show the hand sign for recognition."""

from flask import Flask, render_template, Response, url_for
import cv2
import numpy as np
import os
from keras.models import load_model
import tensorflow as tf
from gtts import gTTS
global graph
global writer
global letter
from skimage.transform import resize

graph = tf.compat.v1.get_default_graph()

# Loading the Model

model = load_model('Train_Model.h5')
vals = ['A','B','C','D','E','F','G','H','I']

#Getting the flask app ready
app = Flask(__name__)
print("[INFO] accessing video stream...")

#Creating a video loader.
def gen():
    global letter
    vid = cv2.VideoCapture(0) #Video object

    
    roi_start = (50,100)
    roi_end = (250, 350)
    vid.set(cv2.CAP_PROP_AUTOFOCUS,0)
    vid.set(3,1280)
    vid.set(4,720)

    
    while (True):
        ret, frame = vid.read() # Reading the video
        frame = cv2.resize(frame, (640,480))
    
        cv2.rectangle(frame,(80,80),(280,280),color=(255,255,255),thickness=5) #Detection Area
        frame1 = frame.copy()
        copy = frame1[80:280, 80:280]
        gray_image = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
        gray_image = cv2.flip(gray_image,1)

        frame1 = mask(copy)
        #Detecting and displaying the output
        writer = detect(frame1)
        cv2.putText(frame,'The Predicted Alphabet is: '+str(writer),(100,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

        #Text to voice converter
        language = 'en'
        voice_out = gTTS(text=writer, lang=language, tld='com')
        voice_out.save(r"C:\Users\Damodharan\Documents\Final_Deliverables\Final-Code\static\aud\{}.mp3".format(writer))
        os.system(r"C:\Users\Damodharan\Documents\Final_Deliverables\Final-Code\static\aud\{}.mp3".format(writer))


        #Displaying the image
        _, col_img = cv2.imencode('.jpg',frame)
        col_img = cv2.flip(col_img,1)
        col_binary = col_img.tobytes()
        col_data_frame = b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + col_binary + b'\r\n\r\n'
        yield col_data_frame

#Masking the frame to detect hand       
def mask(frame):
    gray = frame[:,:,2]
    ret, thresh_gray = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
    thresh_gray = cv2.morphologyEx(thresh_gray, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
    return thresh_gray

#Detecting the alphabet   
def detect(frame):
    global prediction, letter
    img = resize(frame,(64,64,3))
    img = np.expand_dims(img,axis=0)
    if (np.max(img)>1):
        img = img/255.0
    model = load_model('Train_Model.h5')
    prediction = model.predict(img)
    prediction = np.argmax(prediction,axis=1)
    letter = vals[prediction[0]]
    return letter
    

#Loading the page
@app.route('/')
def index():
    return render_template('index.html') #Loading the html file

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype = 'multipart/x-mixed-replace; boundary=frame') #Video Frame

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug = True)
