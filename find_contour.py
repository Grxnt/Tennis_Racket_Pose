import numpy as np
import cv2
from skimage.feature import canny

img = cv2.imread('pog.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,127,255,0)
contours,hierarchy = cv2.findContours(thresh, 1, 2)
cnt = contours[0]

x,y,w,h = cv2.boundingRect(cnt)
img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

ellipse = cv2.fitEllipse(cnt)
img = cv2.ellipse(img,ellipse,(0,0,255),2)

cv2.imwrite("result2.jpg", img)