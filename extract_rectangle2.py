import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import measure
from skimage import morphology as mph
from sklearn.cluster import KMeans

def bboxes(regions):
    bb = list()
    for r in regions:
        x0 = min(r[:,0])
        x1 = max(r[:,0])
        y0 = min(r[:,1])
        y1 = max(r[:,1])
        bb.append((x0,y0,x1,y1))
    return bb

def extract_rectangle(filename, to_filename):
    img = plt.imread(filename)
    #edges = cv2.Canny(img,100,200)
    edges = plt.imread("Output/features.tif")
    
    lines = cv2.HoughLines(edges,1,np.pi/360,200)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 2000*(-b))
        y1 = int(y0 + 2000*(a))
        x2 = int(x0 - 2000*(-b))
        y2 = int(y0 - 2000*(a))
    
        cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)
    
    cv2.imwrite(to_filename+"-lines.tif",img)
    cv2.imwrite(to_filename+'-edges.tif', edges)
    
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
    
img = plt.imread('Photos/160827-112225--NA052964478RU-A77VCD01.tif');
edges = cv2.Canny(img,100,200)

mph.remove_small_objects(edges, min_size=64, connectivity=1, in_place=True)
cv2.imwrite("morph.tif", edges)


c1 = cv2.getTickCount()
labels, num = measure.label(edges, return_num = True)
c2 = cv2.getTickCount()
print "Labeling", c2-c1

c1 = cv2.getTickCount()
cardinals = np.zeros(num+1,dtype=np.int32)

#x = 0
#while x < labels.shape[0]:
#    y = 0
#    while y < labels.shape[1]:
#        l = labels[x,y]
#        cardinals[l] = cardinals[l] + 1
#        y = y + 1
#    x = x + 1
for i in range(num+1):
    m = cv2.compare(labels,int(i), cv2.CMP_EQ)
    cardinals[i] = int(cv2.sumElems(m)[0]/255)
    
sort_inds = np.argsort(cardinals)

regs = list()
for i in range(5):
    regs.append(sort_inds[num-i])
c2 = cv2.getTickCount()
print "Get 10 biggest elements", c2-c1

c1 = cv2.getTickCount()
mask = np.zeros(labels.shape,dtype=np.uint8)
#for x in range(labels.shape[0]):
#    for y in range(labels.shape[1]):
#        if labels[x,y] in regs:
#            mask[x,y] = 255
#        else:
#            mask[x,y] = 0
for r in regs:
    m = cv2.compare(labels, int(r), cv2.CMP_EQ)
    mask = cv2.add(mask, m)
c2 = cv2.getTickCount()
print "Construct mask", (c2-c1)/cv2.getTickFrequency()

c1 = cv2.getTickCount()
result = cv2.bitwise_and(edges, mask)
c2 = cv2.getTickCount()
print "Mask small elements", c2-c1

cv2.imwrite('Output/features.tif', result)

c1 = cv2.getTickCount()
extract_rectangle('Photos/160827-112225--NA052964478RU-A77VCD01.tif',"Output/rect")
c2 = cv2.getTickCount()

print "Extract rectangle", c2-c1