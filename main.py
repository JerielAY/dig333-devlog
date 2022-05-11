import cv2
import numpy as np

# Enable camera
frameWidth = 640
frameHeight = 400
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

# import cascade file for facial recognition
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

#Import cascade file for body recognition 
bodyCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
'''
    # if you want to detect any object for example eyes, use one more layer of classifier as below:
    eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")
'''


def empty(a):
        pass

cv2.namedWindow("HSV")
cv2.resizeWindow("HSV", 640, 240)
cv2.createTrackbar("HUE Min","HSV", 0, 179,empty)
cv2.createTrackbar("HUE Max","HSV", 179, 179,empty)
cv2.createTrackbar("SAT Min","HSV", 0, 255,empty)
cv2.createTrackbar("SAT Max","HSV", 255, 255,empty)
cv2.createTrackbar("VALUE Min","HSV", 0, 255,empty)
cv2.createTrackbar("VALUE Max","HSV", 255, 255,empty)



while True:
    success, img = cap.read()
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Getting corners around the face
    faces = faceCascade.detectMultiScale(imgGray, 1.3, 5)  # 1.3 = scale factor, 5 = minimum neighbor
    # drawing bounding box around face
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # detecting fullbody
    body = bodyCascade.detectMultiScale(imgGray, 1.04, 2)
    # drawing bounding box for fullbody
    for (bx, by, bw, bh) in body:
        img = cv2.rectangle(img, (bx, by), (bx+bw, by+bh), (0, 255, 255), 2)
    
    '''
    # detecting eyes
    eyes = eyeCascade.detectMultiScale(imgGray)
    # drawing bounding box for eyes
    for (ex, ey, ew, eh) in eyes:
        img = cv2.rectangle(img, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 3)
    '''
    #Color detection code
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("HUE Min", "HSV")
    h_max = cv2.getTrackbarPos("HUE Max", "HSV")
    s_min= cv2.getTrackbarPos("SAT Min","HSV")
    s_max= cv2.getTrackbarPos("SAT Max","HSV")
    v_min = cv2.getTrackbarPos("VALUE Min","HSV")
    v_max = cv2.getTrackbarPos("VALUE Max","HSV")



    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHsv, lower, upper)
    result = cv2.bitwise_and(img, img, mask = mask)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    hStack = np.hstack([img, mask, result])

    # cv2.imshow('Original', img)
    # cv2.imshow('HSV Color Space', imgHsv)
    # cv2.imshow('Mask', mask)
    # cv2.imshow('Horizontal Stacking', hStack)

    cv2.imshow('Result', result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # cv2.imshow('face_detect', img)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyWindow('face_detect')