import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
from usefull_functions import *
from asprise_ocr_api import *

filename2 = "Photos/160827-112231--NA052964359RU-A77VCD01.tif"
filename3 = "Photos/160827-112227--NA052964504RU-A77VCD01.tif"
filename4 = "Photos/160827-112255--NA052964230RU-A77VCD01.tif"
filename5 = "Photos/160827-112634--NA052964393RU-A77VCD01.tif"
filename6 = "Photos/160827-112253--NA052964265RU-A77VCD01.tif"
filename7 = "Photos/160827-095711--NA040374451RU-A77VCD01.tif"
filename8 = "Photos/160827-112227--NA052964504RU-A77VCD01-20-.tif"

img = plt.imread(filename2)


Ocr.set_up() # one time setup
ocrEngine = Ocr()
ocrEngine.start_engine("eng")
s = ocrEngine.recognize("Filtered/4.tif", -1, -1, -1, -1, -1,
                  OCR_RECOGNIZE_TYPE_TEXT, OCR_OUTPUT_FORMAT_PLAINTEXT)
print "Result: " + s
# recognizes more images here ..
ocrEngine.stop_engine()