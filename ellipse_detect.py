import matplotlib.pyplot as plt
import numpy as np
import cv2

from skimage import data, color, img_as_ubyte
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter

# Load picture, convert to grayscale and detect edges
img = cv2.imread("images/racket_tilt.jpg")#data.coffee()[0:220, 160:420]
#image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
scale_percent = 300 / len(img)# percent of original size
width = int(img.shape[1] * scale_percent)
height = int(img.shape[0] * scale_percent)
dim = (width, height)
  
# resize image
img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

gray = img[:,:,0]*0.1 + img[:,:,1]*0.8 + img[:,:,2]*0.1
gray = np.uint8(gray)
gray = cv2.GaussianBlur(gray,(5,5),5)
edges = canny(gray)
cv2.imwrite('images/edges.jpg', 255*np.uint8(edges))


# Perform a Hough Transform
# The accuracy corresponds to the bin size of a major axis.
# The value is chosen in order to get a single high accumulator.
# The threshold eliminates low accumulators
result = hough_ellipse(edges, accuracy=20, threshold=250,
                       min_size=80, max_size=120)
result.sort(order='accumulator')

# Estimated parameters for the ellipse
best = list(result[-1])
yc, xc, a, b = [int(round(x)) for x in best[1:5]]
orientation = best[5]

# Draw the ellipse on the original image
cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
img[cy, cx] = (0, 0, 255)
# Draw the edge (white) and the resulting ellipse (red)
edges = color.gray2rgb(img_as_ubyte(edges))
edges[cy, cx] = (250, 0, 0)

fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4),
                                sharex=True, sharey=True)

ax1.set_title('Original picture')
ax1.imshow(img)

ax2.set_title('Edge (white) and result (red)')
ax2.imshow(edges)

plt.show()