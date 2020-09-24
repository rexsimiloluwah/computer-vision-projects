# Applying the face mask detector to real-time video streams


import time
import numpy as np
import os
import cv2
import imutils
from imutils.video import VideoStream
import argparse
from tensorflow.python.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img

#Loading the serialized face mask classifier model
model = load_model("../models/model.h5")

# Loading the serialized face detector CAFFE model from the disk ./face_detector

prototxtPath = os.path.join("../face_detector","deploy.prototxt")
weightsPath = os.path.join("../face_detector", "res10_300x300_ssd_iter_140000.caffemodel")


def detect_face_and_mask(frame):
    faces, locs, preds = [], [], []
    (height, width) = frame.shape[:2]

    face_net = cv2.dnn.readNet(prototxtPath, weightsPath)

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
        (104.0, 177.0, 123.0))

    # Computing face detections from the
    print("Computing face detections from the image ...")
    face_net.setInput(blob)
    detections = face_net.forward()

    #Iterating through all the detections
    for i in range(detections.shape[2]):
        # Extracting the confidence / likelihood of a face being detected
        confidence = detections[0,0,i,2]
        # Filtering out weak detections
        if confidence > 0.7:
            # Compute the x and y co-ordinates for the bounding box of the face
            box = detections[0,0,i, 3:7] * np.array([width,height,width,height])
            (startX, startY, endX, endY) = box.astype("int")

            # Ensuring that the bounding boxes dimensions are within the image frame
            startX, startY = max(0, startX), max(0, startY)
            endX, endY = min(endX, width -1), min(endY, height - 1)

            # Extracting the face from the image based on the bounding boxes
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224,224))
            faces.append(face)
            locs.append([startX, startY, endX, endY])

        # Check if any face was detected
    if len(faces) > 0:
        faces = np.expand_dims((np.array(face)/255.0).astype(float), axis=0)
        preds = model.predict(faces, batch_size = 32)

    return (locs, preds)



vs = VideoStream(src = 0).start()
# Give the camera sensor some time to warm up
# time.sleep(2.0)

# loop over the frames from the video stream
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    (locs, preds) = detect_face_and_mask(frame)
    for (box, pred) in zip(locs, preds):
        (startX, startY, endX, endY) = box
        print(pred, box)
        label = "Masked" if pred[1] <= 0.5  else "Not Masked"
        color = (0, 255, 0) if label == "Masked" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(pred[0], pred[1]) * 100)
        cv2.putText(frame, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
