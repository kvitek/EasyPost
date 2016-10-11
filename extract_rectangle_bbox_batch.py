import cv2
import numpy as np
from matplotlib import pyplot as plt
from usefull_functions import get_edges

filename2 = "Photos/160827-112231--NA052964359RU-A77VCD01.tif"
filename3 = "Photos/160827-112227--NA052964504RU-A77VCD01.tif"
filename4 = "Photos/160827-112255--NA052964230RU-A77VCD01.tif"
filename5 = "Photos/160827-112634--NA052964393RU-A77VCD01.tif"
filename6 = "Photos/160827-112253--NA052964265RU-A77VCD01.tif"
filename7 = "Photos/160827-112227--NA052964504RU-A77VCD01-20-.tif"

def get_main_image(img, crop_width=3):
    crop_img = img[crop_width:img.shape[0]-crop_width, crop_width:img.shape[1]-crop_width]
    edges = get_edges(crop_img, img.shape[1])
    
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
    
    dst = cv2.warpAffine(crop_img,M,(b,a))
    return dst

from os import walk

i = 1
for (dirpath, dirnames, filenames) in walk("Photos"):
    for filename in filenames:
        to_filename = "0000" + str(i)
        to_filename = "Output/" + to_filename[len(to_filename)-5:] + ".tif"
        img = plt.imread(dirpath + "/" + filename)
        dst = get_main_image(img)
        cv2.imwrite(to_filename, dst)
        i=i+1
