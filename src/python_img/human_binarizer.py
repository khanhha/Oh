import os
import cv2 as cv

dir_path = 'D:\Projects\Oh\data\images\prid_450s\cam_a'
out_dir_path = 'D:\Projects\Oh\data\images\prid_450s\cam_a_silhouette'
entries = os.listdir(dir_path)
img_paths = list()
for entry in entries:
    if 'man_' in entry:
        img_paths.append(entry)
        #if len(img_paths) > 10:
        #    break

for file in img_paths:
    file_path = os.path.join(dir_path, file)
    img_sil = cv.imread(file_path)
    img_sil = cv.cvtColor(img_sil, cv.COLOR_BGR2GRAY)
    ret2,img_sil = cv.threshold(img_sil,0,255,cv.THRESH_BINARY)
    img_sil = cv.resize(img_sil,(64,128))
    cv.imwrite(os.path.join(out_dir_path, file), img_sil)




