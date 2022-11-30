import numpy as np
import matplotlib.pyplot as plt
import cv2

# load the image
img = cv2.imread("images/racket_90.jpg")

# convert BGR to RGB to be suitable for showing using matplotlib library
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.medianBlur(gray,5)

thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,11,2)
cv2.imwrite("thresh.jpg", thresh)

# convert image to grayscale
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# apply a blur using the median filter
gray = cv2.medianBlur(gray, 5)

# finds the circles in the grayscale image using the Hough transform
circles = cv2.HoughCircles(image=gray, method=cv2.HOUGH_GRADIENT, dp=0.9, 
                            minDist=80, param1=110, param2=39, maxRadius=500)

for co, i in enumerate(circles[0, :], start=1):
    # draw the outer circle in green
    cv2.circle(img,(int(i[0]),int(i[1])),int(i[2]),(0,255,0),2)
    # draw the center of the circle in red
    cv2.circle(img,(int(i[0]),int(i[1])),2,(0,0,255),3)

cv2.imwrite("ovals.jpg", img)