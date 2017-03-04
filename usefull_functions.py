import cv2
import math
from skimage import measure
from skimage import morphology as mph
import numpy as np
from numpy.linalg import inv

def get_edges(img, threshold):
    edges = cv2.Canny(img,60,200)
    
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
    
def line_distances_axis(lines, precision):
    lines_h = list()
    dist_h = list()
    lines_v = list()
    dist_v = list()
    
    for x1, y1, x2, y2 in lines:
        cos = float(abs(x2-x1))/np.sqrt( (x2-x1)**2+(y2-y1)**2 )
        sin = float(abs(y2-y1))/np.sqrt( (x2-x1)**2+(y2-y1)**2 )
        
        if cos>1-precision:
            lines_h.append((x1,y1,x2,y2))
            dist_h.append(float(y1+y2)/2)
        elif sin > 1-precision:
            lines_v.append((x1,y1,x2,y2))
            dist_v.append(float(x1+x2)/2)
    
    return lines_h, dist_h, lines_v, dist_v
    
def get_text_boxes(img, minSize=5, maxSize=50, h_overlap=2.0, v_overlap=2.0, log=False):
    edges = cv2.Canny(img, 100, 200)
    contours, hierarchy = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
    color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mask = np.ndarray(edges.shape, edges.dtype)
    
    if log: print "Total contours,",len(contours)
    
    i=0
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
                rect2 = (rect[0], (rect[1][0]*v_overlap, rect[1][1]*h_overlap), rect[2])
                box2 = np.array(cv2.cv.BoxPoints(rect2), np.int32)
                cv2.fillPoly(mask,[box2],255)
                i = i+1
    if log: print "Text contours,", i
    
    contours, hierarchy = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
        
    if log: print "Detected boxes,", len(contours)
        
    i=0
    boxes = list()
    b_ceters = list()
    for c in contours:
        rect = cv2.minAreaRect(c)
        if not (rect[1][0]>100 or rect[1][1]>100):
            continue
     
        box = np.array(cv2.cv.BoxPoints(rect), np.int32)
        boxes.append(box)
        box = box.reshape((-1,1,2))
        cv2.polylines(color,[box], True, (0,255,0), 3)
        i = i+1
    
    if log: print "Text boxes,", i
        
    return boxes, color

def get_textbox_centers(img, minSize=5, maxSize=50):
    edges = cv2.Canny(img, 100, 200)
    contours, hierarchy = cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    
    rho = 0.0
    b_centers = list()    
    
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
                b_centers.append(rect[0])
                sz = max(rect[1][:])
                if sz > rho:
                    rho = sz
                
    return b_centers, rho   
    