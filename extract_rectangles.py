import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans



img = plt.imread("file.tif")
edges = cv2.Canny(img,100,200)
lines = cv2.HoughLines(edges,1,np.pi/360,250)

#try to find main angles
km = KMeans(2)
X = lines[0][:,1]
X = X.reshape(-1, 1)
centers = km.fit_predict(X)

theta0 = km.cluster_centers_[1][0]
theta1 = km.cluster_centers_[0][0]

#divide all lines on 2 class
side_a = list()
side_b = list()

i = 0
for c in centers:
    if c == 0:
        side_a.append(lines[0][i,0])
    else:
        side_b.append(lines[0][i,0])
    i = i+1


#bound rectangle lines
sides = list()
sides.append([max(side_a), theta0])
sides.append([min(side_a), theta0])
sides.append([max(side_b), theta1])
sides.append([min(side_b), theta1])

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

cv2.imwrite("file1-lines.tif", img)

A = np.ndarray((2,2))
A[0,0] = np.cos(theta0)
A[0,1] = np.sin(theta0)
A[1,0] = np.cos(theta1)
A[1,1] = np.sin(theta1)

A_inv = np.linalg.inv(A)

b = np.ndarray((2,1))
b[0] = sides[0][0]
b[1] = sides[2][0]
p1 = np.dot(A_inv, b)

b[1] = sides[3][0]
p2 = np.dot(A_inv, b)

b[0] = sides[1][0]
b[1] = sides[2][0]
p3 = np.dot(A_inv, b)

print p1, p2, p3

rows = img.shape[0]
cols = img.shape[1]
a = int(np.sqrt(np.sum(np.multiply(p1-p2,p1-p2))))
b = int(np.sqrt(np.sum(np.multiply(p1-p3,p1-p3))))

pts1 = np.float32([p3.T,p1.T,p2.T])
pts2 = np.float32([[b,0],[0,0],[0,a]])

M = cv2.getAffineTransform(pts1,pts2)

dst = cv2.warpAffine(img,M,(b,a))

cv2.imwrite("rect.tif", dst)
