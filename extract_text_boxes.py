import cv2
from matplotlib import pyplot as plt
from usefull_functions import get_text_boxes

from os import walk

i = 1
for (dirpath, dirnames, filenames) in walk("Photos"):
    for filename in filenames:
        to_filename = "0000" + str(i)
        to_filename = "Output/" + to_filename[len(to_filename)-5:] + ".tif"
        img = plt.imread(dirpath + "/" + filename)
        dst = get_text_boxes(img, maxSize=25, h_overlap=2.0, v_overlap=2.0)
        cv2.imwrite(to_filename, dst)
        i=i+1
        print filename, i