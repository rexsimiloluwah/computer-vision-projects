import cv2
import imutils
from imutils.video import VideoStream

vs = VideoStream(src = 0).start()
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break


cv2.destroyAllWindows()
vs.stop()