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
    
def get_trans_points(lines_0, lines_1):
    dim0 = len(lines_0)
    dim1 = len(lines_0)
    b = np.zeros((dim0*dim1, 1))
    A = np.zeros((dim0*dim1, dim0*dim1))
    j = 0
    for line0 in lines_0:
        for line1 in lines_1:
            A[j,j] = line0[3]-line0[1]
            A[j,j+1] = -line0[2]+line0[0]
            b[j] = -line0[1]*line0[2] + line0[0]*line0[3]
            
            A[j+1,j] = line1[3]-line1[1]
            A[j+1,j+1] = -line1[2]+line1[0]
            b[j+1] = -line1[1]*line1[2] + line1[0]*line1[3]
            
            j = j+2
    A_inv = inv(A)
    X = np.dot(A_inv,b)
    
    return np.int32(X)