from pickle import TRUE
import cv2

# Enable camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 420)

while TRUE:
    success, frame = cap.read()
    for i in range(2):
        cv2.imwrite('opencv'+str(i)+'.png', frame)


#Gray conversion and noise reduction (smoothening)
gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
gray_frame=cv2.GaussianBlur(gray_frame,(25,25),0)

#Where I will create the baseline image 


#Calculating the difference and image thresholding
delta=cv2.absdiff(baseline_image,gray_frame)
threshold=cv2.threshold(delta,35,255, cv2.THRESH_BINARY)[1]
# Finding all the contours
(contours,_)=cv2.findContours(threshold,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Drawing rectangles bounding the contours (whose area is > 5000)
for contour in contours:
    if cv2.contourArea(contour) < 5000:
        continue
    (x, y, w, h)=cv2.boundingRect(contour)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 1)


del(frame)
