import cv2
from skimage import measure
from skimage import morphology as mph
import numpy as np
from numpy.linalg import inv

def get_edges(img, threshold):
    edges = cv2.Canny(img,100,200)
    
    labels, num = measure.label(edges, return_num = True)
    mph.remove_small_objects(labels, min_size=threshold, connectivity=2, in_place=True)
    res = cv2.compare(labels, 0, cv2.CMP_GT)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    res = cv2.morphologyEx(res,cv2.MORPH_CLOSE,kernel)
    res = res > 0
    skl = mph.skeletonize(res)
    
    skl = skl.astype('uint8')*255
    
    return skl
    
def get_corners(edges, threshold):
    dst = np.float32(edges)
    dst = cv2.cornerHarris(dst,3,3,threshold)
    return dst
    
def line_distances(x0, y0, theta, lines):
    X_1 = np.array(lines[:,0:2])
    X_2 = np.array(lines[:,2:4])
    X_1 = X_1.T
    X_2 = X_2.T
    m = X_1.shape[1]
    p = np.ndarray((2,1), dtype=float)
    p[0] = x0; p[1] = y0;
    P = np.multiply(p, np.ones((2,m)))
    
    v = np.zeros((2,1))
    v[1] = np.sin(theta); v[0] = np.cos(theta);
    
    Y_1 = X_1 - P
    Y_2 = X_2 - P
    res_1 = Y_1 - np.dot(v, np.dot(v.T, Y_1))
    res_2 = Y_2 - np.dot(v, np.dot(v.T, Y_2))
    res_1 = res_1**2
    res_2 = res_2**2    
    res = np.sum(res_1, axis=0) + np.sum(res_2,axis=0)
    res = np.sqrt(res)
    return res
    