import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
from skimage.segmentation import relabel_sequential
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian
import os
import pdb

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

img = cv2.imread("./data_densecrf/2020-10-27_10:16:28.png")
img = cv2.resize(img, (1280, 720))
mask_core = cv2.imread("./data_densecrf/mask_core.png", cv2.IMREAD_ANYDEPTH)
mask_core = cv2.resize(mask_core, (1280, 720), cv2.INTER_NEAREST)
mask_core = (mask_core/np.max(mask_core)).astype(np.uint32)

mask_around = cv2.imread("./data_densecrf/mask_around.png", cv2.IMREAD_ANYDEPTH)
mask_around = cv2.resize(mask_around, (1280, 720), cv2.INTER_NEAREST)
mask_around = (mask_around/np.max(mask_around)).astype(np.uint32)

image_width = img.shape[1]
image_height = img.shape[0]
annos = np.zeros([image_height, image_width], dtype=np.uint32)
#mask_around = np.zeros([image_height, image_width], dtype=np.uint32)
#mask_around = mask_around + 1
annos += mask_around
#kernel = np.ones((5,5),np.uint8)
#annos += cv2.dilate(mask_core.astype(np.uint8), kernel, iterations=3)
#annos -= cv2.bitwise_and(mask_around.astype(np.uint8), mask_core.astype(np.uint8))
annos += mask_core * 2
colors, labels = np.unique(annos, return_inverse=True)

EPSILON = 1e-8

n_label = 3
tau = 1.05
# Setup the CRF model
d = dcrf.DenseCRF2D(image_width, image_height, n_label)

#anno_norm = annos / 255.
#mask_foreground = mask_core/255.
#n_energy = -np.log((1.0 - mask_foreground + EPSILON)) / (tau * sigmoid(1 - mask_foreground))
#p_energy = -np.log(mask_foreground + EPSILON) / (tau * sigmoid(mask_foreground))
#U = np.zeros((n_label, img.shape[0] * img.shape[1]), dtype='float32')
#U[0, :] = n_energy.flatten()
#U[1, :] = p_energy.flatten()
U = unary_from_labels(labels, n_label, gt_prob=0.90, zero_unsure=True)
d.setUnaryEnergy(U)

d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

# This adds the color-dependent term, i.e. features are (x,y,r,g,b).
d.addPairwiseBilateral(sxy=(20, 20), srgb=(5, 50, 13), rgbim=img,
                        compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

# Find out the most probable class for each pixel.
Q, tmp1, tmp2 = d.startInference()
for i in range(20):
    print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
    d.stepInference(Q, tmp1, tmp2)


res = np.argmax(Q, axis=0)
res = res.reshape(img.shape[:2])
res[res == 1] = 0
res = res*255
cv2.imwrite("res.png", res.astype('uint8'))
cv2.waitKey(10)