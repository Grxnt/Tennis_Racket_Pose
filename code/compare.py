import cv2
from contours import getBoundingBox
from detection import detect

filename_1 = 'images/control.jpg'
filename_2 = 'images/out_more.jpg'
img_1 = cv2.imread(filename_1)
img_2 = cv2.imread(filename_2)

# do this once
cnr_1 = detect(filename_1)
cnr_2 = detect(filename_2)
cv2.imwrite('cnr_1.jpg', cnr_1)
cv2.imwrite('cnr_2.jpg', cnr_2)

# then import the saved variables
# cnr_1 = cv2.imread('cnr_1.jpg')
# cnr_2 = cv2.imread('cnr_2.jpg')
# cnr_1 = cv2.cvtColor(cnr_1, cv2.COLOR_BGR2GRAY)
# cnr_2 = cv2.cvtColor(cnr_2, cv2.COLOR_BGR2GRAY)

box_1 = getBoundingBox(cnr_1)
box_2 = getBoundingBox(cnr_2)

img_1 = cv2.drawContours(img_1,[box_1],0,(0,0,255),2)
img_2 = cv2.drawContours(img_2,[box_2],0,(0,0,255),2)

cv2.imwrite('box1.jpg', img_1)
cv2.imwrite('box2.jpg', img_2)

transformation_matrix = cv2.getPerspectiveTransform(box_1.astype('float32'), 
                                                    box_2.astype('float32'))
angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(transformation_matrix)
