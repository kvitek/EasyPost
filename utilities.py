import numpy as np
import cv2
from matplotlib import pyplot as plt
from usefull_functions import get_edges
from usefull_functions import line_distances

img = plt.imread("Output/00003.tif")
w = np.where(img>0)
points = np.array(zip(w[1], w[0]))

rect = cv2.minAreaRect(points)
box = cv2.cv.BoxPoints(rect)
box = np.int0(box)
cv2.drawContours(img,[box],0,(255),2)
cv2.line(img, (int(rect[0][0]),int(rect[0][1])), (int(rect[1][0]),int(rect[1][1])), (255), 2)

cv2.imwrite("Filtered/bbox.tif", img)


