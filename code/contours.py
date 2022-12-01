import cv2
import numpy as np
from skimage.feature import canny

def getBoundingBox(img):
    ## contour detection
    ret,thresh = cv2.threshold(img,127,255,0)
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]
    
    x,y,w,h = cv2.boundingRect(cnt)
    img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
    
    ellipse = cv2.fitEllipse(cnt)
    img = cv2.ellipse(img,ellipse,(0,255,0),2)
    
    cv2.imwrite('result.jpg', img)
    
    ## filling in the object
    new_img = np.zeros_like(img)
    new_img = cv2.ellipse(new_img,ellipse,255,2)
    thresh = cv2.threshold(new_img, 128, 255, cv2.THRESH_BINARY)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)
    result = np.zeros_like(new_img)
    cv2.drawContours(result, [big_contour], 0, 255, cv2.FILLED)
    
    ## box drawing
    new_img = result
    ret,thresh = cv2.threshold(new_img,127,255,0)
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    cnt = contours[0]
    
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    new_img = cv2.drawContours(new_img,[box],0,255,2)
    
    cv2.imwrite('images/oval.jpg', new_img)
    return box
