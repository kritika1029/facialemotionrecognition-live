# from typing import Reversible
from django.shortcuts import redirect, render
# from fer.codes import startCam
# from django.http import *
from tensorflow.keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
import ctypes 

stop=False

def index(request):
    return render(request, 'index.html')

def startcamera(request):
    face_classifier = cv2.CascadeClassifier(r'static/fer/haarcascade_frontalface_default.xml')
    #classifier =load_model(r'C:/Users/kritiaro/Documents/fer/Emotion_Detection_CNN/model.h5')
    #classifier =load_model(r'static/fer/positive-surprise.h5')
    classifier=load_model(r'static/fer/positive-surprise.h5')

    # emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']
    emotion_labels = ['negative','positive']

    global cap 
    cap = cv2.VideoCapture(0)
    c=0
    framecount=0
    print(type(cap))

    while(not stop):
        _, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)
        framecount+=1
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
            
            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)
                prediction = classifier.predict(roi)[0]
                label=emotion_labels[prediction.argmax()]
                if(label=='negative'):
                    c+=1
                    print(c)
                if(c>=60):
                    c=0
                    ctypes.windll.user32.MessageBoxW(0, "Negative Emotion Recognised", "❌ ALERT ❌", 1)
                   
                # label=prediction.argmax()
                label_position = (x,y)
                if(framecount>=100):
                    print("FRAMECOUNT:::",framecount)
                    framecount=0
                    c=0
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            else:
                cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow('Emotion Detector- Press "q" or "Stop Camera" button to exit',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

    return redirect('home')

def stopcamera(request):
    stop=True
    global cap 
    cap.release()
    cv2.destroyAllWindows()
    return redirect('home')

# if request.method == 'POST' and 'run_script' in request.POST:
#     startCam()
#     return HttpResponseRedirect(reverse(''))

# def startcam(request):
#     if request.method == 'POST' and 'run_script' in request.POST:
#         startCam()
#         return render(request, 'index.html',)


# Create your views here.
