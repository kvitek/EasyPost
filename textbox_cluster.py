import cv2
import math
import numpy as np
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from usefull_functions import get_textbox_centers


filename2 = "Photos/160827-112231--NA052964359RU-A77VCD01.tif"
filename3 = "Photos/160827-112227--NA052964504RU-A77VCD01.tif"
filename4 = "Photos/160827-112255--NA052964230RU-A77VCD01.tif"
filename5 = "Photos/160827-112634--NA052964393RU-A77VCD01.tif"
filename6 = "Photos/160827-112253--NA052964265RU-A77VCD01.tif"
filename7 = "Photos/160827-095711--NA040374451RU-A77VCD01.tif"
filename8 = "Photos/160827-112227--NA052964504RU-A77VCD01-20-.tif"

img = plt.imread(filename7)
color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

centers, rho = get_textbox_centers(img, maxSize=30)
X = np.array(centers)

db = DBSCAN(eps=rho*1.5, min_samples=5).fit(X)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)

unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
colors = colors*255
colors = np.int32(colors)

for k, col in zip(unique_labels, colors):
    if k == -1:
        continue

    class_member_mask = (labels == k)

    xy = np.int32(X[class_member_mask & core_samples_mask])
    for x in xy:
        cv2.circle(color, (x[0],x[1]), int(rho/2), (col[0],col[1],col[2]), 1)
        
cv2.imwrite("Filtered/clusters.tif", color)
