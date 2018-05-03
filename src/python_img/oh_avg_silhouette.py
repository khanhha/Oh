import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

dir = 'D:\Projects\Oh\data\images\silhouette\silhouette'
sub_dir_names = list()
for root, dirs, files in os.walk(dir, topdown=True):
    for name in dirs:
       sub_dir_names.append(name)

target_file_names = {}
for dir_name in sub_dir_names:
    dir_path= os.path.join(dir, dir_name)
    file_paths = list()
    for file_name in os.listdir(dir_path):
        file_paths.append(os.path.join(dir_path, file_name))
    target_file_names[dir_name] = file_paths


path = 'D:\Projects\Oh\data\images\silhouette\\avg_silhouette_map'
for key, value in target_file_names.items():
    imgs = list()
    for file_path in value:
        img = cv.cvtColor(cv.imread(file_path), cv.COLOR_BGRA2GRAY)
        ret, img = cv.threshold(img, 1, 255, cv.THRESH_BINARY)
        img = cv.morphologyEx(img, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_RECT,(4,4)))
        img = img.astype(np.float32)/255
        imgs.append(img)

    avg_img = np.zeros(imgs[0].shape, dtype = np.float32)
    for img in imgs:
        avg_img += img
    avg_img /= len(imgs)
    avg_img *= 255
    avg_img = avg_img.astype(np.uint8)
    cv.imwrite(os.path.join(path, str(key)+'.png'), avg_img)

