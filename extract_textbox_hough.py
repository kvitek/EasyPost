import cv2
import math
from skimage import measure
from skimage import morphology as mph
import numpy as np
from matplotlib import pyplot as plt

filename2 = "Photos/160827-112231--NA052964359RU-A77VCD01.tif"
filename3 = "Photos/160827-112227--NA052964504RU-A77VCD01.tif"
filename4 = "Photos/160827-112255--NA052964230RU-A77VCD01.tif"
filename5 = "Photos/160827-112634--NA052964393RU-A77VCD01.tif"
filename6 = "Photos/160827-112253--NA052964265RU-A77VCD01.tif"
filename7 = "Photos/160827-095711--NA040374451RU-A77VCD01.tif"
filename8 = "Photos/160827-112227--NA052964504RU-A77VCD01-20-.tif"

def get_mask(img, minSize=5, maxSize=50):
    edges = cv2.Canny(img, 100, 200)
    contours, hierarchy = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
    mask = np.ndarray(edges.shape, edges.dtype)
    
    for c in contours:
        x,y,h,w = cv2.boundingRect(c)
        if h<100 and w<100:
            rect = cv2.minAreaRect(c)
            if rect[1][1]>maxSize or rect[1][0]>maxSize:
                continue
            if rect[1][1] < 0.1:
                continue
            if rect[1][1] < minSize and rect[1][0] < minSize:
                continue
            
            aspect = rect[1][0]/rect[1][1]
            if aspect<3 and aspect>0.3:
                hull = cv2.convexHull(c)                
                cv2.fillPoly(mask,[hull],255)
                
    
    return mask

img = plt.imread(filename2)
mask = get_mask(img)
color = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)

cv2.imwrite("Filtered/mask.tif",color)