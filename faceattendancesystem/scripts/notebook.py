#!/usr/bin/env python
# coding: utf-8

# ## Installing and Importing the Required packages

# In[1]:


get_ipython().system('pip freeze > requirements.txt')


# In[3]:


get_ipython().system('pip install cmake')


# In[4]:


get_ipython().system('pip install tqdm')


# In[5]:


# Text to speech module
get_ipython().system('pip install pyttsx3')


# In[1]:


import numpy as np
import cv2
import imutils
import os
import re
import pandas as pd
import time
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import face_recognition
from imutils.video import VideoStream


# In[2]:


# Testing the Python Speech synthesis module

import pyttsx3
engine = pyttsx3.init()
start = time.time()
engine.say("Please look directly into the Webcam to enroll your face !")
engine.runAndWait()
end = time.time()
print(end - start)


# In[3]:


def init_voice(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()


# In[5]:


init_voice("I am Blessed !")


# ### Setting up directories

# In[4]:


if not os.path.isdir("models"):
    os.mkdir("models")
    
if not os.path.isdir("images"):
    os.mkdir("images")

if not os.path.isdir("outputs"):
    os.mkdir("outputs")

if not os.path.isdir("tests"):
    os.mkdir("tests")
    
if not os.path.isdir("scripts"):
    os.mkdir("scripts")


# ## Let's get started

# ### Loading the serialized CAFFE model for frontal face detection from the disk

# In[5]:


# Declaring paths
BASE_MODEL_DIR = "./models/frontalface_detector"

prototxtPath = os.path.join(BASE_MODEL_DIR,"deploy.prototxt")
weightsPath = os.path.join(BASE_MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")


# In[6]:


# Loading the model
face_net = cv2.dnn.readNet(prototxtPath, weightsPath)


# In[7]:


# Reading a random image

img_path = "./tests/showdemcamp.jpg"
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
img_copy = img.copy()

(height, width) = img.shape[:2]
cv2.imshow("Random Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()


# ### Running the face detection model

# In[8]:


# Construct a blob from the Image
start = time.time()
blob = cv2.dnn.blobFromImage(img,
                             1.0,
                             (300, 300),
                             (104.0, 177.0,123.0) )

print("[PROCESS] :- Computing detections from image")
face_net.setInput(blob)
detections = face_net.forward()
end = time.time()
print("-----"*10)
print(f"Runtime elapsed :- {end - start} ms")


# In[9]:


detections.shape


# In[10]:


# Iterating through all detections 

for i in range(detections.shape[2]):
    # Extracting the confidence/likelihood of the detection
    confidence = detections[0,0,i,2]
    # Filtering out weak detections
    if confidence > 0.9:
        # Compute the x and y co-ordinates for the bounding box of the face
        box = detections[0,0,i, 3:7] * np.array([width,height,width,height])
        
        (startX, startY, endX, endY) = box.astype("int")

        # Ensuring that the bounding boxes dimensions are within the image frame
        startX, startY = max(0, startX), max(0, startY)
        endX, endY = min(endX, width -1), min(endY, height - 1)

        # Extracting the face from the image based on the bounding boxes
        face = img[startY:endY, startX:endX]
        color = (255,0,0)
        label = f"{np.round(confidence*100, 2)}%"
        cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
        cv2.putText(img, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
plt.figure(figsize = (12,12))
plt.imshow(img)
plt.show()


# In[11]:


# Helper function for detecting faces from images

def detect_face(frame):
    (height, width) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame,
                             1.0,
                             (300, 300),
                             (104.0, 177.0,123.0) )

    print("[PROCESS] :- Computing detections from image")
    face_net.setInput(blob)
    detections = face_net.forward()
    
    for i in range(detections.shape[2]):
        # Extracting the confidence/likelihood of the detection
        confidence = detections[0,0,i,2]
        # Filtering out weak detections
        if confidence > 0.9:
            # Compute the x and y co-ordinates for the bounding box of the face
            box = detections[0,0,i, 3:7] * np.array([width,height,width,height])

            (startX, startY, endX, endY) = box.astype("int")

            # Ensuring that the bounding boxes dimensions are within the image frame
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(endX, width -1), min(endY, height - 1)

            # Extracting the face from the image based on the bounding boxes
            face = frame[startY:endY, startX:endX]
            color = (255,255,0)
            label = f"{np.round(confidence*100, 2)}%"
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
            cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    


# ### Detecting faces from Video Streams for Face enrolment

# In[12]:


# Initialize the camera sensor, and allow the sensor to warm up for 2s
print("[PROCESS] :- Initializing the Video Stream")
videostream = VideoStream(src = 0).start()
time.sleep(2.0)
init_voice("Please enter your Name to get started (firstname_lastname)")
name = str(input("Please enter your Name :-"))

pattern = re.compile("^([a-zA-Z])+_([a-zA-Z])+")
if not re.match(pattern, name):
    init_voice("Please enter a valid name (firstname_lastname) or press q to exit")
    name = str(input("Please enter your Name (firstname_lastname) :-"))
    if name == "q":
        print("[EXITING] :- Thanks for trying out the system.")
        sys.exit(0)

person_dir = os.path.join("./images", name)
if not os.path.isdir(person_dir):
    os.mkdir(person_dir)

n = 0
# Looping over the frames
while True:
    
    # Grabbing the frame from the videostream
    frame = videostream.read()
    orig = frame.copy()
    # Resizing the frame
    frame = imutils.resize(frame, width = 500)
    
    detect_face(frame)
    cv2.putText(frame, f"You have saved {n}/10 time(s)", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
            
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # if the 's' key is pressed, save the image to the directory
    if key == ord("s"):
        path = os.path.join(person_dir, f"{str(n).zfill(3)}.jpg")
        cv2.imwrite(path, orig)
        n += 1
        
    # if the `q` key was pressed, break from the loop
    if key == ord("q") or n >= 10:
        break
        
cv2.destroyAllWindows()
videostream.stop()
    


# In[13]:


encodings = {}
for p in os.listdir("images"):
    if not os.listdir(os.path.join("images",p)):
        pass
    
    pic = face_recognition.load_image_file(f"./images/{p}/000.jpg")
    pic_encoding = face_recognition.face_encodings(pic)[0]
    
    encodings[p] = pic_encoding  


# In[14]:


encodings


# ### Recognition

# In[18]:


# Initialize the camera sensor, and allow the sensor to warm up for 2s
print("[PROCESS] :- Initializing the Video Stream")
videostream = VideoStream(src = 0).start()
time.sleep(2.0)
name = ""
detected = False
# Looping over the frames
while True:
    
    # Grabbing the frame from the videostream
    frame = videostream.read()
    orig = frame.copy()
    # Resizing the frame
    frame = imutils.resize(frame, width = 500)
    
    detect_face(frame)
    
    #print(face_encodings)
    
    if not detected:
        face_locations = face_recognition.face_locations(frame)
        if (face_locations):
            face_encodings = face_recognition.face_encodings(frame, face_locations)[0]
        for k,v in encodings.items():
            print(face_recognition.compare_faces( [face_encodings], np.array(v), tolerance = 0.4)[0])
            if face_recognition.compare_faces( [face_encodings], np.array(v), tolerance = 0.4)[0] == True:
                name = k
                init_voice(f"Welcome, {k}. How may i help you ?")
                detected = True
                break
    
    name = "".join(name.split("_"))
    cv2.putText(frame, f"Welcome, {name}!!", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
        
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
        
cv2.destroyAllWindows()
videostream.stop()
    


# In[16]:


moyo = face_recognition.load_image_file("./images/moyosore_okunowo/000.jpg")
moyo_encoding = face_recognition.face_encodings(moyo)[0]


# In[64]:


simi = face_recognition.load_image_file("./images/similoluwa_okunowo/000.jpg")
simi_encoding = face_recognition.face_encodings(simi)[0]


# In[65]:


face_recognition.compare_faces([simi_encoding], moyo_encoding, tolerance=0.5)


# In[ ]:




