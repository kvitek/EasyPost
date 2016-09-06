import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
from skimage import morphology as mph
from sklearn.cluster import KMeans

def get_edges(img, threshold):
    edges = cv2.Canny(img,100,200)
    
    labels, num = measure.label(edges, return_num = True)
    res = mph.remove_small_objects(labels, min_size=threshold, connectivity=2, in_place=False)
    res = cv2.compare(res, 0, cv2.CMP_GT)
    
    return res

filename1 = "output/00002-edges.tif"
filename2 = "Photos/160827-112231--NA052964359RU-A77VCD01.tif"
filename3 = "Photos/160827-112227--NA052964504RU-A77VCD01.tif"
filename4 = "Photos/160827-112255--NA052964230RU-A77VCD01.tif"
filename5 = "Photos/160827-112634--NA052964393RU-A77VCD01.tif"

img = plt.imread(filename5)
#edges = get_edges(img, 500)

col, row = img.shape
col = int(col/2)
row = int(row/2)
f = np.fft.fft2(img, s = (col, row))

fshift = np.fft.fftshift(f)
magnitude_spectrum = 20*np.log(np.abs(fshift))

res = cv2.compare(magnitude_spectrum, 200, cv2.CMP_GT)
cv2.imwrite("Filtered/spectrum.tif", res)

#erase artifacts
#cv2.line(res, (0,int(col/2)), (row, int(col/2)), (0), 1)
#cv2.line(res, (int(row/2),0), (int(row/2), col), (0), 1)
c = (int(row/2), int(col/2))
cv2.circle(res, c, 30, (0), thickness=-1)


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
morph = cv2.morphologyEx(res,cv2.MORPH_OPEN,kernel)

inds = np.where(morph>0)
X = np.array(zip(inds[0],inds[1]), dtype=float)

km = KMeans(4)

km.fit(X)
c = (int(row/2), int(col/2))
color = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
for i,j in km.cluster_centers_:
    cv2.line(color,c,(int(j),int(i)),(0,255,0),1)
    
cv2.imwrite("Filtered/spectrum.tif", res)
cv2.imwrite("Filtered/spectrum_m.tif", color)
#edges = cv2.Canny(morph, 100, 200)

#lines = cv2.HoughLines(res,1,np.pi/180,500)
##
#for rho, theta in lines[0]:
#     a = np.cos(theta)
#     b = np.sin(theta)
#     x0 = a*rho
#     y0 = b*rho
#     x1 = int(x0 + 3000*(-b))
#     y1 = int(y0 + 3000*(a))
#     x2 = int(x0 - 3000*(-b))
#     y2 = int(y0 - 3000*(a))
#     
#     cv2.line(res,(x1,y1),(x2,y2),(0),1)

#minLineLength = 100
#maxLineGap = 10
#lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
#for x1,y1,x2,y2 in lines[0]:
#    cv2.line(img,(x1,y1),(x2,y2),(0),2)

