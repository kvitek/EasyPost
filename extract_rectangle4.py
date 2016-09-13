import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift
from usefull_functions import get_edges
from usefull_functions import line_distances

filename2 = "Photos/160827-112231--NA052964359RU-A77VCD01.tif"
filename3 = "Photos/160827-112227--NA052964504RU-A77VCD01.tif"
filename4 = "Photos/160827-112255--NA052964230RU-A77VCD01.tif"
filename5 = "Photos/160827-112634--NA052964393RU-A77VCD01.tif"
filename6 = "Photos/160827-112253--NA052964265RU-A77VCD01.tif"


img = plt.imread(filename6)

edges = get_edges(img, 2000)
cv2.imwrite("Filtered/skl_edges.tif", edges)

minLineLength = 120
maxLineGap = 10
color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
angles = np.zeros((len(lines[0]),1))
i = 0
for x1,y1,x2,y2 in lines[0]:
    #cv2.line(color,(x1,y1),(x2,y2),(0,255,0),1)
    angles[i] = math.asin((y2-y1)/np.sqrt((x2-x1)**2+(y2-y1)**2))
    i = i+1

ms = MeanShift(bandwidth=np.pi/40)
labels = ms.fit_predict(angles)
print ms.cluster_centers_/np.pi*180

a_counts = list()
for c in range(ms.cluster_centers_.shape[0]):
    a_counts.append((np.where(labels==c)[0].shape[0],c))

a_counts.sort(reverse=True)

print a_counts

lines0 = lines[0][np.where(ms.labels_==a_counts[0][1])]
lines1 = lines[0][np.where(ms.labels_==a_counts[1][1])]
theta0 = ms.cluster_centers_[a_counts[0][1]]
theta1 = ms.cluster_centers_[a_counts[1][1]]

for x1,y1,x2,y2 in lines0:
    cv2.line(color,(x1,y1),(x2,y2),(255,0,0),2)
for x1,y1,x2,y2 in lines1:
    cv2.line(color,(x1,y1),(x2,y2),(0,255,0),2)
 
epsilon = 15
   
dists = line_distances(0, 0, theta0, lines0)


ms_lines = MeanShift(bandwidth=epsilon)
ms_lines.fit(dists.reshape((dists.shape[0],1)))

h_lines = np.ndarray((ms_lines.cluster_centers_.shape[0], 4), dtype=int)
for c in range(ms_lines.cluster_centers_.shape[0]):
    l = lines0[np.where(ms_lines.labels_ == c)]
    mi = min(l[:,0]); ma = max(l[:,2])
    p0 = l[np.where(l[:,0]==mi),0:2]
    p1 = l[np.where(l[:,2]==ma),2:4]
    h_lines[c,0:2] = p0[0]
    h_lines[c,2:4] = p1[0]
    print c
    print dists[np.where(ms_lines.labels_ == c)]

dists = line_distances(0, 0, theta1, lines1)

ms_lines = MeanShift(bandwidth=epsilon)
ms_lines.fit(dists.reshape((dists.shape[0],1)))

v_lines = np.ndarray((ms_lines.cluster_centers_.shape[0], 4), dtype=int)
for c in range(ms_lines.cluster_centers_.shape[0]):
    l = lines1[np.where(ms_lines.labels_ == c)]
    mi = min(l[:,0]); ma = max(l[:,2])
    p0 = l[np.where(l[:,0]==mi),0:2]
    p1 = l[np.where(l[:,2]==ma),2:4]
    v_lines[c,0:2] = p0[0]
    v_lines[c,2:4] = p1[0]
    
for x1,y1,x2,y2 in h_lines:
    cv2.line(color,(x1,y1),(x2,y2),(0,0,255),2)
for x1,y1,x2,y2 in v_lines:
    cv2.line(color,(x1,y1),(x2,y2),(0,0,255),2)

cv2.imwrite("Filtered/morph_edges.tif", color)