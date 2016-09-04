import cv2
import numpy as np
from sklearn.cluster import KMeans

img = cv2.imread('packet.tif')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,100,200,apertureSize = 3)

lines = cv2.HoughLines(edges,1,np.pi/360,250)

km = KMeans()

centers = km.fit_predict(lines[0])

for rho,theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 2000*(-b))
    y1 = int(y0 + 2000*(a))
    x2 = int(x0 - 2000*(-b))
    y2 = int(y0 - 2000*(a))

    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)

cv2.imwrite('houghlines3.jpg',img)
cv2.imwrite('edges.tif', edges)