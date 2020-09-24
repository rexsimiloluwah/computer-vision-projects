import numpy as np
import os
import cv2
import argparse
from tensorflow.python.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img

#Loading the serialized face mask classifier model
model = load_model("../models/model.h5")

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help="Path to Image")
args = vars(ap.parse_args())

# Loading the serialized face detector CAFFE model from the disk ./face_detector

prototxtPath = os.path.join("../face_detector","deploy.prototxt")
weightsPath = os.path.join("../face_detector", "res10_300x300_ssd_iter_140000.caffemodel")

face_net = cv2.dnn.readNet(prototxtPath, weightsPath)

img = cv2.imread(args["image"])
img_copy = img.copy()
(height, width) = img.shape[:2]

# construct a blob from the image
blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300),
	(104.0, 177.0, 123.0))

# Computing face detections from the
print("Computing face detections from the image ...")
face_net.setInput(blob)
detections = face_net.forward()
# print(detections, detections.shape)

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
        face = img[startY:endY, startX:endX]
        face = cv2.resize(face, (224,224), interpolation = cv2.INTER_AREA)
        face = np.expand_dims((face/255.0).astype(float), axis=0)
        pred = model.predict(face)
        # print(pred.reshape(1,-1)[0][1])
        label = "Masked : {}%".format(np.round(pred[0][0]*100,2)) if pred[0][1] <= 0.5 else "Not Masked : {}%".format(np.round(pred[0][1]*100,2))
        color = (0,255,0) if pred[0][1] <= 0.5 else (0,0,255)
        cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
        cv2.putText(img, label, (startX, startY -10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)



cv2.imwrite("../outputs/output_{}".format(args["image"].split("/")[-1]), img)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

