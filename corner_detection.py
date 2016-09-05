import cv2
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

filename1 = 'output/00002-edges.tif'
filename2 = "Photos/160827-112227--NA052964504RU-A77VCD01.tif"
img = plt.imread(filename1)


gray = np.float32(img)
dst = cv2.cornerHarris(gray,3,3,0.04)

#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)

# Threshold for an optimal value, it may vary depending on the image.
color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
color[dst>0.3*dst.max()]=[0,0,255]

r = np.transpose(np.nonzero(dst>0.3*dst.max()))
km = KMeans(2)
X = lines[0][:,1]
X = X.reshape(-1, 1)
centers = km.fit_predict(X)

cv2.imwrite('Output/corners.tif',color)
