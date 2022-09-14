import cv2  # OpenCV

import numpy as np

def drawbox(object,frame):
    """
    Drawing bouding box around `object` on `frame`
    """
    img = frame
    for (x,y,w,h) in object:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    return img

# Enable webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Cannot open camera")

cap.set(3,640) # width
cap.set(4,480) # height


# Import cascade (face detection)
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret,frame = cap.read()
    if not ret:
        raise Exception("Can't receive frame")
        break

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)   # grayscale mode

    # Getting corners around the face
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)  # 1.3 = scale factor, 5 = minimum neighbor can be detected

    # Drawing bouding box around face
    frame = drawbox(faces,frame)

    # Displaying image with box
    cv2.imshow('face_detect',frame)

    # cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):   # quit when hit 'q'
        break

# Release capture
cap.release()
cv2.destroyAllWindows()
