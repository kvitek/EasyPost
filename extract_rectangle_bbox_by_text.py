import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from sklearn.cluster import MeanShift
from usefull_functions import get_text_boxes

filename2 = "Photos/160827-112231--NA052964359RU-A77VCD01.tif"
filename3 = "Photos/160827-112227--NA052964504RU-A77VCD01.tif"
filename4 = "Photos/160827-112255--NA052964230RU-A77VCD01.tif"
filename5 = "Photos/160827-112634--NA052964393RU-A77VCD01.tif"
filename6 = "Photos/160827-112253--NA052964265RU-A77VCD01.tif"
filename7 = "Photos/160827-095711--NA040374451RU-A77VCD01.tif"
filename8 = "Photos/160827-112227--NA052964504RU-A77VCD01-20-.tif"

img = plt.imread(filename4)

boxes = get_text_boxes(img, maxSize=25)

ts = list()
W = 0
H = 0
for b in boxes:
    c = b[0]-b[1]
    w = np.sqrt(np.dot(c,c))
    c = b[2]-b[1]
    h = np.sqrt(np.dot(c,c))
    srcp = np.ndarray((3,2), dtype=np.float32)
    dstp = np.ndarray((3,2), dtype=np.float32)
    if w>h:
        tmp = h
        h = w
        w = tmp
        srcp[0] = b[3]
        srcp[1] = b[0]
        srcp[2] = b[1]
    else:
        srcp[0] = b[0]
        srcp[1] = b[1]
        srcp[2] = b[2]
    dstp[0] = [0,w-1]
    dstp[1] = [0,0]
    dstp[2] = [h-1,0]
    
    ts.append([int(w),int(h),srcp,dstp])
    if h>H: H = int(h)
    W = W + int(w)

out = np.zeros((W,H), dtype=img.dtype)
p = 0
for t in ts:
    M = cv2.getAffineTransform(t[2], t[3])
    dst = cv2.warpAffine(img, M, (t[1],t[0]))
    out[p:p+t[0], 0:t[1]] = dst[:,:]
    p = p + t[0]
    print p, t[1]
    
cv2.imwrite("Filtered/out.tif", out)
    
