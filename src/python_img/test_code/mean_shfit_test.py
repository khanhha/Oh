import cv2 as cv
import os
from common import util
import matplotlib.pyplot as plt

DIR = 'D:/Projects/Oh/data/images/silhouette/images/view_152/'
OUT_DIR = 'D:/Projects/Oh/data/images/silhouette/mean_shift_result/'

spatialRad = 20
colorRad = 10
maxPyrLevel = 1

for file_name in os.listdir(DIR):
    file_path = f'{DIR}{file_name}'
    img = cv.imread(file_path)
    img = util.resize_common_size(img)
    seg_img = cv.pyrMeanShiftFiltering(img, sp=spatialRad, sr = colorRad, maxLevel = 0)
    plt.imshow(seg_img)
    plt.show()
