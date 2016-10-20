import cv2
import numpy as np
from matplotlib import pyplot as plt


filename2 = "Photos/160827-112231--NA052964359RU-A77VCD01.tif"
filename3 = "Photos/160827-112227--NA052964504RU-A77VCD01.tif"
filename4 = "Photos/160827-112255--NA052964230RU-A77VCD01.tif"
filename5 = "Photos/160827-112634--NA052964393RU-A77VCD01.tif"
filename6 = "Photos/160827-112253--NA052964265RU-A77VCD01.tif"
filename7 = "Photos/160827-095711--NA040374451RU-A77VCD01.tif"
filename8 = "Photos/160827-112227--NA052964504RU-A77VCD01-20-.tif"

img = plt.imread(filename3)
blur = cv2.GaussianBlur(img,(5,5),0)
#ret3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
cv2.imwrite("Filtered/binary.tif", th3)