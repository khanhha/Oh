"""
Adapted from the inference.py to demonstate the usage of the util functions.
"""

import sys
import os
import cv2 as cv
import numpy as np
import pydensecrf.densecrf as dcrf
import matplotlib.pyplot as plt

from common import background_subtractor as bgsub
from common import util as cutil
from common import prob_map

# Get im{read,write} from somewhere.
try:
    from cv2 import imread, imwrite
except ImportError:
    # Note that, sadly, skimage unconditionally import scipy and matplotlib,
    # so you'll need them if you don't have OpenCV. But you probably have them.
    from skimage.io import imread, imsave
    imwrite = imsave
    # TODO: Use scipy instead.

from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

# if len(sys.argv) != 4:
#     print("Usage: python {} IMAGE ANNO OUTPUT".format(sys.argv[0]))
#     print("")
#     print("IMAGE and ANNO are inputs and OUTPUT is where the result should be written.")
#     print("If there's at least one single full-black pixel in ANNO, black is assumed to mean unknown.")
#     sys.exit(1)
#
# fn_im = sys.argv[1]
# fn_anno = sys.argv[2]
# fn_output = sys.argv[3]

DIR = 'D:\Projects\Oh\data\images\crf_test'
IMG_DIR = 'D:\Projects\Oh\data\images\silhouette\images\\view_178'
PROB_MAP_DIR = 'D:\Projects\Oh\data\images\silhouette\\avg_silhouette_map'
img_label_mapping = 'D:\Projects\Oh\data\images\silhouette\mapping_image_avg_silhouette.txt'

#fn_im = f'{DIR}\\im4.png'
fn_anno = f'{DIR}\\anno4.png'
#fn_output = f'{DIR}\\out4.png'


#build background model
bg_img_path = "D:\Projects\Oh\data\images\\trainim_LL_0428\\test\\image178_empty_LL_0321.jpg"
bg_img = cutil.preprocess_img(cv.imread(bg_img_path))
bg_model = bgsub.build_background_model(bg_img)
for img_name in os.listdir(IMG_DIR):
    fn_im = f'{IMG_DIR}\{img_name}'
    img = cutil.preprocess_img(cv.imread(fn_im))
    fg_mask = bgsub.extract_foreground_mask(bg_model, img)

for img_name in os.listdir(IMG_DIR):
    fn_im = f'{IMG_DIR}\{img_name}'
    fn_output = f'D:\Projects\Oh\data\images\crf_test\output\{img_name}'
    print('start processing image {img_name}')
    print('\t\t do background subtraction')
    img = cutil.preprocess_img(cv.imread(fn_im))
    fg_mask_sub = bgsub.extract_foreground_mask(bg_model, img)
    fg_mask_sub = cv.erode(fg_mask_sub, cv.getStructuringElement(cv.MORPH_RECT, (22,22)))
    fg_mask_sub = fg_mask_sub.astype(np.bool)

    print('\t\t do mean silhouette')
    prb_label = prob_map.find_mean_silhouette_label(img_label_mapping, img_name)
    prb_img = prob_map.load_mean_silhouette(PROB_MAP_DIR, prb_label)
    fg_mask, bg_mask, mask = prob_map.build_fg_bg_masks(prb_img, False, fg_threshold=0.8, as_bool = False)
    fg_mask = cutil.resize_common_size(fg_mask)
    bg_mask = cutil.resize_common_size(bg_mask)
    fg_mask = fg_mask.astype(bool)
    bg_mask = bg_mask.astype(bool)

    # plt.subplot(1,3,1); plt.imshow(fg_mask)
    # plt.subplot(1,3,2); plt.imshow(fg_mask_sub)
    # plt.subplot(1,3,3); plt.imshow(np.bitwise_or(fg_mask, fg_mask_sub))
    # plt.show()

    fg_mask  = np.bitwise_or(fg_mask, fg_mask_sub)

    print('\t\t do crf')
    anno_rgb = np.zeros_like(img, dtype=np.uint8)
    anno_rgb[fg_mask] = (255,0,0)
    anno_rgb[bg_mask] = (0, 0, 255)
    # plt.subplot(1,2,1)
    # plt.imshow(anno_rgb)
    # plt.subplot(1,2,2)
    # plt.imshow(img)
    # plt.show()

    anno_rgb = anno_rgb.astype(np.int32)
    # Convert the annotation's RGB color to a single 32-bit integer color 0xBBGGRR
    anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)

    # Convert the 32bit integer color to 1, 2, ... labels.
    # Note that all-black, i.e. the value 0 for background will stay 0.
    colors, labels = np.unique(anno_lbl, return_inverse=True)

    # But remove the all-0 black, that won't exist in the MAP!
    HAS_UNK = 0 in colors
    if HAS_UNK:
        print("Found a full-black pixel in annotation image, assuming it means 'unknown' label, and will thus not be present in the output!")
        print("If 0 is an actual label for you, consider writing your own code, or simply giving your labels only non-zero values.")
        colors = colors[1:]
    #else:
    #    print("No single full-black pixel found in annotation image. Assuming there's no 'unknown' label!")

    # And create a mapping back from the labels to 32bit integer colors.
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16

    # Compute the number of classes in the label image.
    # We subtract one because the number shouldn't include the value 0 which stands
    # for "unknown" or "unsure".
    n_labels = len(set(labels.flat)) - int(HAS_UNK)
    print(n_labels, " labels", (" plus \"unknown\" 0: " if HAS_UNK else ""), set(labels.flat))

    ###########################
    ### Setup the CRF model ###
    ###########################
    use_2d = False
    # use_2d = True
    if use_2d:
        print("Using 2D specialized functions")

        # Example using the DenseCRF2D code
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,
                               compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
    else:
        print("Using generic 2D functions")

        # Example using the DenseCRF class and the util functions
        d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=HAS_UNK)
        d.setUnaryEnergy(U)

        # This creates the color-independent features and then add them to the CRF
        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=3,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This creates the color-dependent features and then add them to the CRF
        feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)


    ####################################
    ### Do inference and compute MAP ###
    ####################################

    # Run five inference steps.
    Q = d.inference(5)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at first.
    MAP = colorize[MAP,:]
    MAP = MAP.reshape(img.shape)

    plt.subplot(1, 3, 1)
    plt.imshow(img[:,:,::-1])
    plt.subplot(1, 3, 2)
    plt.imshow(img[:,:,::-1])
    plt.imshow(anno_rgb.astype(np.uint8), alpha=0.2)
    plt.subplot(1, 3, 3)
    plt.imshow(img[:,:,::-1])
    plt.imshow(MAP, alpha=0.2)
    plt.savefig(fn_output, dpi = 1000)
    #imwrite(fn_output, MAP.reshape(img.shape))

    # Just randomly manually run inference iterations
    Q, tmp1, tmp2 = d.startInference()
    for i in range(5):
        print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
        d.stepInference(Q, tmp1, tmp2)