import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


img = cv2.imread('file1.tif',0)
edges = cv2.Canny(img,100,200)

lines = cv2.HoughLines(edges,1,np.pi/360,250)

print lines[0].shape
normalization = raw_input("Normalization y/n:")

if normalization.lower() == "n":
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 3000*(-b))
        y1 = int(y0 + 3000*(a))
        x2 = int(x0 - 3000*(-b))
        y2 = int(y0 - 3000*(a))

        cv2.line(img,(x1,y1),(x2,y2),(0),1)
else:
    km = KMeans(2)
    X = lines[0][:,1]
    X = X.reshape(-1, 1)
    centers = km.fit_predict(X)
    side_a = list()
    side_b = list()
    
    i = 0
    for c in centers:
        if c == 0:
            side_a.append(lines[0][i,0])
        else:
            side_b.append(lines[0][i,0])
        i = i+1
    
    km_s = KMeans(2)
    X = np.array(side_a).reshape(-1,1)
    km_s.fit_predict(X)
    
    sides = list()
    for c in km_s.cluster_centers_:
        sides.append([c,km.cluster_centers_[0]])
    
    X = np.array(side_b).reshape(-1,1)
    km_s.fit_predict(X)
    for c in km_s.cluster_centers_:
        sides.append([c,km.cluster_centers_[1]])
        
    for rho, theta in sides:
         a = np.cos(theta)
         b = np.sin(theta)
         x0 = a*rho
         y0 = b*rho
         x1 = int(x0 + 3000*(-b))
         y1 = int(y0 + 3000*(a))
         x2 = int(x0 - 3000*(-b))
         y2 = int(y0 - 3000*(a))
         
         cv2.line(img,(x1,y1),(x2,y2),(0),2)

cv2.imwrite('file1-lines.tif',img)