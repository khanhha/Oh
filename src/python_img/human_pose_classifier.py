import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import skimage.feature as feature
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report
from sklearn import metrics

label_file = open('D:\Projects\Oh\data\images\prid_450s\cam_a_label.txt')
imgs= list()
img_labels = list()
for line in label_file.readlines():
    sub_strs = line.split()
    imgs.append(cv.imread(sub_strs[0]))
    img_labels.append(int(sub_strs[1]))

y = np.ravel(img_labels)

for index, img in enumerate(imgs):
    img = cv.resize(img, (96, 160))
    img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
    imgs[index] = img

#pre-process
# clahe = cv.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
# for index, img in enumerate(imgs):
#     img_1 = clahe.apply(img)
#     imgs[index] = img_1
#     plt.subplot(1,2,1); plt.imshow(img,   cmap = 'gray')
#     plt.subplot(1,2,2); plt.imshow(img_1, cmap = 'gray')
#     plt.show()
# exit(10)


X = np.array([feature.hog(img) for img in imgs])
print(X.shape)
print(y.shape)

#parameters are estimated in the code human_pose_estimator_params
clf = svm.SVC(kernel='rbf', C=1000, gamma = 0.0001)
#score = cross_val_score(estimator=clf, X= X, y = y, cv=5)
#print(score)
clf.fit(X,y)

dir = 'D:\Projects\Oh\data\images\prid_450s\cam_b'
hogs_in = []
img_paths = []
for file_name in os.listdir(dir):
    if 'img_' in file_name:
        file_path = os.path.join(dir, file_name)
        img = cv.imread(file_path)
        img = cv.resize(img, (96, 160))
        img = cv.cvtColor(img, cv.COLOR_BGRA2GRAY)
        img_hog = feature.hog(img)
        hogs_in.append(img_hog)
        img_paths.append(file_path)

labels = clf.predict(hogs_in)

# dir_out = 'D:\Projects\Oh\data\images\prid_450s\cam_b_classify_result'
# for index, img_path in enumerate(img_paths):
#     label = labels[index]
#     file_name = os.path.basename(img_path)
#     path_out = os.path.join(dir_out, str(label))
#     try:
#         os.stat(path_out)
#     except:
#         os.mkdir(path_out)
#     path_out = os.path.join(os.path.join(path_out, file_name))
#     cv.imwrite(path_out, cv.imread(img_path))

file_out = 'D:\Projects\Oh\data\images\prid_450s\cam_b_classify_result.txt'
file = open(file_out,'w')
for index, img_path in enumerate(img_paths):
    label = labels[index]
    file.write(img_path + "\t" +str(label)+'\n')
file.close()



















