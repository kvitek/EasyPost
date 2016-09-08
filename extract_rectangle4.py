import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from skimage import measure
from sklearn.cluster import MeanShift
from usefull_functions import get_edges
from usefull_functions import get_corners
from usefull_functions import get_trans_points

filename2 = "Photos/160827-112231--NA052964359RU-A77VCD01.tif"
filename3 = "Photos/160827-112227--NA052964504RU-A77VCD01.tif"
filename4 = "Photos/160827-112255--NA052964230RU-A77VCD01.tif"
filename5 = "Photos/160827-112634--NA052964393RU-A77VCD01.tif"


img = plt.imread(filename2)

edges = get_edges(img, 1000)
cv2.imwrite("Filtered/skl_edges.tif", edges)

minLineLength = 100
maxLineGap = 10
color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
angles = np.zeros((len(lines[0]),1))
i = 0
for x1,y1,x2,y2 in lines[0]:
    cv2.line(color,(x1,y1),(x2,y2),(0,255,0),1)
    angles[i] = math.acos(abs(y1-y2)/np.sqrt((x2-x1)**2+(y2-y1)**2))
    i = i+1
    
print len(lines[0])

ms = MeanShift(bandwidth=np.pi/40)
ms.fit(angles)

print ms.cluster_centers_ 

lines0 = lines[0][np.where(ms.labels_==0)]
lines1 = lines[0][np.where(ms.labels_==1)]
lines2 = lines[0][np.where(ms.labels_==2)]
for x1,y1,x2,y2 in lines0:
    cv2.line(color,(x1,y1),(x2,y2),(255,0,0),2)
for x1,y1,x2,y2 in lines1:
    cv2.line(color,(x1,y1),(x2,y2),(0,255,0),2)
for x1,y1,x2,y2 in lines2:
    cv2.line(color,(x1,y1),(x2,y2),(0,0,255),2)
    
X = get_trans_points(lines0, lines1)
for i in range(X.shape[0]/2):
    cv2.circle(color,(X[2*i],X[2*i+1]),2,(0,0,255),thickness=-1)

cv2.imwrite("Filtered/morph_edges.tif", color)