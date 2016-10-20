import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift
from usefull_functions import get_edges
from usefull_functions import line_distances
from usefull_functions import line_distances_axis

filename2 = "Photos/160827-112231--NA052964359RU-A77VCD01.tif"
filename3 = "Photos/160827-112227--NA052964504RU-A77VCD01.tif"
filename4 = "Photos/160827-112255--NA052964230RU-A77VCD01.tif"
filename5 = "Photos/160827-112634--NA052964393RU-A77VCD01.tif"
filename6 = "Photos/160827-112253--NA052964265RU-A77VCD01.tif"
filename7 = "Photos/160827-095711--NA040374451RU-A77VCD01.tif"
filename8 = "Photos/160827-112227--NA052964504RU-A77VCD01-20-.tif"

img = plt.imread(filename3)
edges = cv2.Canny(img, 100, 200)
#edges = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
#                cv2.THRESH_BINARY,11,2)
#ret, edges = cv2.threshold(img,127,255,cv2.THRESH_BINARY_INV)

cv2.imwrite("Filtered/edges.tif", edges)
contours, hierarchy = cv2.findContours(edges,cv2.RETR_CCOMP ,cv2.CHAIN_APPROX_SIMPLE)

color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
mask = np.ndarray(edges.shape, edges.dtype)

print "Total contours,",len(contours)
i=0
for c in contours:
    x,y,h,w = cv2.boundingRect(c)
    if h<100 and w<100:
        rect = cv2.minAreaRect(c)
        if rect[1][1] > 50 or rect[1][0]>50:
            continue
        if rect[1][1] < 0.1:
            continue
        if rect[1][1] < 5:
            continue
        if rect[1][0] < 5:
            continue
        aspect = rect[1][0]/rect[1][1]
        if aspect<3 and aspect>0.3:
            rect2 = (rect[0], (rect[1][0]*2, rect[1][1]*2), rect[2])
            box2 = np.array(cv2.cv.BoxPoints(rect2), np.int32)
            box = np.array(cv2.cv.BoxPoints(rect), np.int32)
            #cv2.drawContours(color, [box], 0, (0,255,0), 1)
            box = box.reshape((-1,1,2))
            cv2.polylines(color,[box], True, (0,255,0), 1)
            cv2.fillPoly(mask,[box2],255)
            i = i+1
print "Text contours,", i

cv2.imwrite("Filtered/boxes.tif", color)
cv2.imwrite("Filtered/mask.tif", mask)

contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

print "Detected boxes,", len(contours)
i=0
for c in contours:
    rect = cv2.minAreaRect(c)
    if not (rect[1][0]>100 or rect[1][1]>100):
        continue
 
    box = np.array(cv2.cv.BoxPoints(rect), np.int32)
    box = box.reshape((-1,1,2))
    cv2.polylines(color,[box], True, (0,255,0), 3)
    i = i+1

print "Text boxes,", i
    
cv2.imwrite("Filtered/text_boxes.tif", color)

