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

X = np.array([feature.hog(img) for img in imgs])
print(X.shape)
print(y.shape)

# clf = svm.SVC(kernel='rbf', C=1000, gamma = 0.0001)
# score = cross_val_score(estimator=clf, X= X, y = y, cv=5)
# print(score)
# exit()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0)

tuned_parameters = [
    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
    {'kernel': ['rbf'], 'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
]
clf = GridSearchCV(estimator=svm.SVC(), param_grid=tuned_parameters, cv = 5, scoring='precision_macro')
clf.fit(X_train,y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on development set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print()

print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()





