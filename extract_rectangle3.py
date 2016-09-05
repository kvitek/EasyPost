import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
from skimage import morphology as mph
from sklearn.cluster import KMeans

def extract_rectangle(img, edges, to_filename):
          
    lines = cv2.HoughLines(edges,1,np.pi/360,100)
        
    #try to find main angles
    km = KMeans(2)
    X = lines[0][:,1]
    X = X.reshape(-1, 1)
    centers = km.fit_predict(X)
    
    theta0 = km.cluster_centers_[0][0]
    theta1 = km.cluster_centers_[1][0]
    
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
    if theta0<theta1:
        sides.append([max(side_a), theta0])
        sides.append([min(side_a), theta0])
        sides.append([max(side_b), theta1])
        sides.append([min(side_b), theta1])
    else:
        sides.append([max(side_b), theta1])
        sides.append([min(side_b), theta1])
        sides.append([max(side_a), theta0])
        sides.append([min(side_a), theta0])
        tmp = theta0
        theta0 = theta1
        theta1 = tmp
        
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
    
    a = int(np.sqrt(np.sum(np.multiply(p1-p2,p1-p2))))
    b = int(np.sqrt(np.sum(np.multiply(p1-p3,p1-p3))))
    
    pts1 = np.float32([p3.T,p1.T,p2.T])
    pts2 = np.float32([[b,0],[0,0],[0,a]])
    
    M = cv2.getAffineTransform(pts1,pts2)
    
    dst = cv2.warpAffine(img,M,(b,a))
    
    cv2.imwrite(to_filename+".tif", dst)
    
def get_edges(img, threshold):
    edges = cv2.Canny(img,100,200)
    
    labels, num = measure.label(edges, return_num = True)
    res = mph.remove_small_objects(labels, min_size=threshold, connectivity=2, in_place=False)
    res = cv2.compare(res, 0, cv2.CMP_GT)
    
    return res
    
#main program

from os import walk

i = 1
for (dirpath, dirnames, filenames) in walk("Photos"):
    for filename in filenames:
        to_filename = "0000" + str(i)
        to_filename = to_filename[len(to_filename)-5:]
        
        img = plt.imread(dirpath+"/"+filename)
        edges = get_edges(img, 500)
        cv2.imwrite("output/"+to_filename+"-edges.tif", edges)
        extract_rectangle(img, edges , "output/"+to_filename)
        i=i+1
        print "File:",filename