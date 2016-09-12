import cv2
from matplotlib import pyplot as plt
from os import walk
from usefull_functions import get_edges

i = 1
for (dirpath, dirnames, filenames) in walk("Photos"):
    for filename in filenames:
        to_filename = "0000" + str(i)
        to_filename = to_filename[len(to_filename)-5:]
        img = plt.imread(dirpath+"/"+filename)
        edges = get_edges(img, 2000)
        cv2.imwrite("Output/"+to_filename+".tif", edges)
        i=i+1