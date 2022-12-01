import cv2
from contours import getBoundingBox
from detection import detect

filename1 = 'images/control.jpg'
filename2 = 'images/40_percent.jpg'
img1 = cv2.imread(filename1)
img2 = cv2.imread(filename2)
cnr1 = detect(filename1)
cnr2 = detect(filename2)
#gray1 = cv2.cvtColor(cnr1, cv2.COLOR_BGR2GRAY)
#gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
box1 = getBoundingBox(cnr1)
box2 = getBoundingBox(cnr2)

cnr_1 = cv2.drawContours(img1,[box1],0,(0,0,255),2)
cnr_2 = cv2.drawContours(img2,[box2],0,(0,0,255),2)

cv2.imwrite('box1.jpg', cnr_1)
cv2.imwrite('box2.jpg', cnr_2)
cv2.imwrite('preBox.jpg', cnr1)