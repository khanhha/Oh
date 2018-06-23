import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import skimage.color as skcolor
import skimage.feature as skfeature
import skimage.filters as skfilter
import skimage.exposure as skexposure

import cv2 as cv
import os
from common import util
import scipy

DIR = 'D:/Projects/Oh/data/images/silhouette/images/view_152/'
OUT_DIR = 'D:/Projects/Oh/data/images/silhouette/edge_detection/'

for file_name in os.listdir(DIR):
    file_path = f'{DIR}{file_name}'
    print(file_name)
    img2 = scipy.misc.imread(file_path, mode="L")
    img2 = util.preprocess_img(img2)
    edges1 = skfeature.canny(img2, 0.0)
    edge_roberts = skfilter.roberts(img2)
    edge_sobel = skfilter.sobel(img2)
    edge_sobel = skexposure.equalize_adapthist(edge_sobel, kernel_size=5)
    segment_mask1 = felzenszwalb(img2, scale=20)
    plt.subplot(141), plt.imshow(edges1, cmap='gray')
    plt.subplot(142), plt.imshow(edge_sobel, cmap='gray')
    plt.subplot(143), plt.imshow(edge_roberts, cmap='gray')
    plt.subplot(144), plt.imshow(img2), plt.imshow(skcolor.label2rgb(segment_mask1), alpha=0.4)
    plt.savefig(f'{OUT_DIR}{file_name[:-4]}.png', dpi=1000)
exit()

for file_name in os.listdir(DIR):
    file_path = f'{DIR}{file_name}'
    img2 = scipy.misc.imread(file_path, mode="L")
    segment_mask1 = felzenszwalb(img2, scale=50)
    #segment_mask2 = felzenszwalb(img2, scale=1000)

    plt.subplot(121), plt.imshow(mark_boundaries(img2, segment_mask1))
    plt.subplot(122), plt.imshow(img2), plt.imshow(skcolor.label2rgb(segment_mask1), alpha=0.4)
    #plt.subplot(122);plt.imshow(mark_boundaries(img2, segment_mask2))
    plt.show()