import cv2
import numpy as np
from matplotlib import pyplot as plt

def bboxes(regions):
    bb = list()
    for r in regions:
        x0 = min(r[:,0])
        x1 = max(r[:,0])
        y0 = min(r[:,1])
        y1 = max(r[:,1])
        bb.append((x0,y0,x1,y1))
    return bb
    
img = cv2.imread('Photos/160827-112225--NA052964478RU-A77VCD01.tif', 0);
edges = cv2.Canny(img,100,200)


cv2.imwrite("Output/edges.tif", edges)
mser = cv2.MSER()
mser.setDouble("areaThreshold",2.0)
mser.setInt("maxArea", 300)
mser.setInt("minArea", 10)
mser.setInt("edgeBlurSize", 5)
mser.setInt("delta", 5)

regions = mser.detect(img)
bb = bboxes(regions)

for b in bb:
    cv2.rectangle(img,(b[0],b[1]),(b[2],b[3]),(0),1)

cv2.imwrite('Output/features.tif', img)