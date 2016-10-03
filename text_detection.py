import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift
from usefull_functions import get_edges
from usefull_functions import line_distances

filename2 = "Filtered/text2.tif"

img = plt.imread(filename2)

edges = cv2.Canny(img,100,200)
mser = cv2.MSER()
regions = mser.detect(img)
#kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#edges = cv2.morphologyEx(edges,cv2.MORPH_CLOSE,kernel)

cv2.imwrite("Filtered/edges.tif", edges)

color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
contours, h = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

rects = list()
#for cnt in contours:
#    x,y,w,h = cv2.boundingRect(cnt)
#    area = cv2.contourArea(cnt)
#    solidity = float(area)/(w*h)
#    if float(h)/w < 4.0 and float(h)/w > 0.3 and solidity > 0.01 and h<50 and h>10:
#        cv2.drawContours(color, [cnt], 0, (0,255,0), 1)
#        cv2.rectangle(color, (x,y), (x+w,y+h), (255,0,0), -1)
#        rects.append((x,y,w,h))

for m in regions:
    m = m.reshape(-1, 1, 2)
    x,y,w,h = cv2.boundingRect(m)
    area = cv2.contourArea(m)
    solidity = float(area)/(w*h)
    if float(h)/w < 4.0 and float(h)/w > 0.3 and solidity > 0.1 and h<50 and h>10:
        rects.append((x,y,w,h))        
        cv2.rectangle(color, (x,y), (x+w,y+h), (0,0,255), -1)

g_rects = cv2.groupRectangles(rects,groupThreshold=1,eps=1000.0)
#d_rects = g_rects[0][np.where(g_rects[1]==1)]
for r in g_rects[0]:
    cv2.rectangle(color, (r[0],r[1]), (r[0]+r[2],r[1]+r[3]), (0,0,255), 1)
cv2.imwrite("Filtered/result.tif", color)