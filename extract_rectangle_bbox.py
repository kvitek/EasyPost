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

img = plt.imread(filename7)
edges = get_edges(img, 100)

cv2.imwrite("Filtered/edges.tif", edges)

crop_img = img[]
edges = get_edges(img, 2000)

w = np.where(edges>0)
points = np.array(zip(w[1], w[0]))

rect = cv2.minAreaRect(points)
box = cv2.cv.BoxPoints(rect)
box = np.float_(box)

p1 = box[0]
p2 = box[1]
p3 = box[2]
a = int(np.sqrt(np.sum(np.multiply(p1-p2,p1-p2))))
b = int(np.sqrt(np.sum(np.multiply(p2-p3,p2-p3))))

pts1 = np.float32([p1.T,p2.T,p3.T])
pts2 = np.float32([[0,a],[0,0],[b,0]])

M = cv2.getAffineTransform(pts1,pts2)

dst = cv2.warpAffine(img,M,(b,a))
edges = cv2.warpAffine(edges,M,(b,a))

#kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
#edges = cv2.morphologyEx(edges,kernel,1)

minLineLength = 120
maxLineGap = 10
color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)

lines_h, dist_h, lines_v, dist_v = line_distances_axis(lines[0], 0.1)

epsilon = 15
ms = MeanShift(bandwidth=epsilon)

dists = np.array(dist_h)
ms.fit(dists.reshape((dists.shape[0],1)))

h_base = list()

i = 0
#for l in ms.labels_:
#    ls = lines_h[np.where(ms.labels_==l)]
#    m1 = min(ls[:][0])
#    m2 = min(ls[:][2])
#    if m1<m2: mn =m1
#    else: mn=m2
#    
#    m1 = max(ls[:][1])
#    m2 = max(ls[:][3])
#    if m1>m2: mx =m1
#    else: mx=m2
#    
#    h_base.append(ms.cluster_centers_[i], mn, mx)
#    i = i+1

for x1,y1,x2,y2 in lines_h:
    cv2.line(color,(x1,y1),(x2,y2),(255,0,0),1)
for x1,y1,x2,y2 in lines_v:
    cv2.line(color,(x1,y1),(x2,y2),(0,255,0),1)

cv2.imwrite("Filtered/rect.tif", dst)
cv2.imwrite("Filtered/edges.tif", color)