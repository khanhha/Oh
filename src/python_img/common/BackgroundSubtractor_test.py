import os
import cv2 as cv
import matplotlib.pyplot as plt
from BackgroundSubtractor import BackgroundSubtractor

bg_img_path = "D:\Projects\Oh\data\images\\trainim_LL_0428\\test\\image178_empty_LL_0321.jpg"
IMG_DIR = 'D:\Projects\Oh\data\images\silhouette\images\\view_178'

bgsub = BackgroundSubtractor(bg_img_path)

for img_name in os.listdir(IMG_DIR):
    fn_im = f'{IMG_DIR}\{img_name}'
    img = cv.imread(fn_im)
    masked_img = bgsub.extract_foreground_mask(img)
    plt.imshow(masked_img, cmap='gray')
    plt.show()