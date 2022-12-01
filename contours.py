import cv2
import numpy as np
from skimage.feature import canny

## Load picture, convert to grayscale and detect edges
img = cv2.imread("images/racket_tilt.jpg")#data.coffee()[0:220, 160:420]
#image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
scale_percent = 300 / len(img)# percent of original size
width = int(img.shape[1] * scale_percent)
height = int(img.shape[0] * scale_percent)
dim = (width, height)
  
# resize image
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
# convert to grayscale w preference towards red
gray = img[:,:,0]*0.1 + img[:,:,1]*0.8 + img[:,:,2]*0.1
gray = cv2.GaussianBlur(np.uint8(gray),(5,5),5)
edges = canny(gray, sigma=3)
cv2.imwrite('images/edges.jpg', 255*np.uint8(edges))

## filling in the object
img = cv2.imread('images/edges.jpg', cv2.IMREAD_GRAYSCALE)
hh, ww = img.shape[:2]
# threshold
thresh = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1]
# get the (largest) contour
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
big_contour = max(contours, key=cv2.contourArea)
# draw white filled contour on black background
result = np.zeros_like(img)
cv2.drawContours(result, [big_contour], 0, (255,255,255), cv2.FILLED)
# save results
cv2.imwrite('images/contour.jpg', result)

## contour detection
img = cv2.imread('images/contour.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,127,255,0)
contours,hierarchy = cv2.findContours(thresh, 1, 2)
cnt = contours[0]

x,y,w,h = cv2.boundingRect(cnt)
img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

ellipse = cv2.fitEllipse(cnt)
img = cv2.ellipse(img,ellipse,(0,255,0),2)

cv2.imwrite('result.jpg', img)

## filling in the object
new_img = np.zeros_like(img)
new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
new_img = cv2.ellipse(new_img,ellipse,(255,255,255),2)
thresh = cv2.threshold(new_img, 128, 255, cv2.THRESH_BINARY)[1]
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]
big_contour = max(contours, key=cv2.contourArea)
result = np.zeros_like(new_img)
cv2.drawContours(result, [big_contour], 0, (255,255,255), cv2.FILLED)
cv2.imwrite('images/oval.jpg', result)

## box drawing
new_img = cv2.imread('images/oval.jpg')
gray = cv2.cvtColor(new_img, cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,127,255,0)
contours,hierarchy = cv2.findContours(thresh, 1, 2)
cnt = contours[0]

rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
new_img = cv2.drawContours(new_img,[box],0,(0,0,255),2)

cv2.imwrite('images/oval.jpg', new_img)
