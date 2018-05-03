import cv2 as cv
import matplotlib.pyplot as plt
import os
from skimage.measure import compare_ssim
from skimage import data, img_as_float

dir = "D:\Projects\Oh\data\images\\trainim_LL_0428\\test"
out_dir = "D:\Projects\Oh\data\images\\trainim_LL_0428\\test\\background_subtraction_result"

bg_img = cv.imread("D:\Projects\Oh\data\images\\trainim_LL_0428\\test\image178_empty_LL_0321.jpg")
bg_img = cv.GaussianBlur(bg_img,(3,3),0)
bg_img = cv.cvtColor(bg_img, cv.COLOR_BGR2GRAY)

fgbg = cv.createBackgroundSubtractorMOG2()

for file_name in os.listdir(dir):
    file_path = os.path.join(dir, file_name)
    if os.path.isfile(file_path) and '.jpg' in file_name:
        img = cv.imread(file_path)
        img = cv.GaussianBlur(img, (3, 3),0)
        fgbg.apply(img)

kernel = cv.getStructuringElement(cv.MORPH_RECT,(15,15))
for file_name in os.listdir(dir):
    file_path = os.path.join(dir, file_name)
    if os.path.isfile(file_path) and '.jpg' in file_name:
        img = cv.imread(file_path)
        img = cv.GaussianBlur(img, (3, 3),0)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        fgmask = fgbg.apply(img)
        #fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)
        #plt.imshow(fgmask)
        #plt.show()

        file_name = file_name.rpartition('.')[-3]
        file_name_1 = "cv_{0}.png".format(file_name)
        file_name_2 = "rick_{0}.png".format(file_name)
        file_name_3 = "sk_{0}.png".format(file_name)
        cv.imwrite(os.path.join(out_dir, file_name_1), fgmask)

        #img_dif = cv.absdiff(bg_img, img)
        #cv.imwrite(os.path.join(out_dir, file_name_2), img_dif)

        bg_img = img_as_float(bg_img)
        img = img_as_float(img)
        score, img_dif = compare_ssim(bg_img, img, multichannel=True, data_range=img.max() - img.min(), full=True)
        img_dif = 1.0 - img_dif
        img_dif = (img_dif * 255).astype("uint8")
        #img_dif = cv.cvtColor(img_dif, cv.COLOR_BGR2GRAY)
        cv.imwrite(os.path.join(out_dir, file_name_3), img_dif)



