#!/usr/bin/python
"""
Adapted from the original C++ example: densecrf/examples/dense_inference.cpp
http://www.philkr.net/home/densecrf Version 2.2

Author: Wei Ji
College: The OID Lab, Dalian University of Technology
"""


import numpy as np
import cv2
import pydensecrf.densecrf as dcrf
# from skimage.segmentation import relabel_sequential
import os


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def crf(input_path, sal_path, output_path, binary=0.3):

    root = os.getcwd()

    if not os.path.exists(root + '/' + output_path):
        os.makedirs(root + '/' + output_path)
    output_path = root + '/' + output_path

    files = os.listdir(input_path)
    # os.chdir(input_path)
    roo = os.getcwd()

    total_num = len(files)
    count_num = int(total_num / 10)
    count = 0

    for file in files:
        count += 1
        img = cv2.imread(root + '/' + input_path + '/' + file, 1)
        annos = cv2.imread(root + '/' + sal_path + '/' + file[:-4] + '.png', 0)
        if binary is not None:
            annos[annos < 255*binary] = 0  # 255*0.2=51
            annos[annos >= 255*binary] = 255  # 255*0.4=102   255*0.3=76.5
        # labels = relabel_sequential(cv2.imread(root + '/' + sal_path + '/' + file[:-4] + '.png', 0))[0].flatten()
        output = output_path + '/' + file[:-4] + '.png'

        EPSILON = 1e-8

        M = 2  # salient or not
        tau = 1.05
        # Setup the CRF model
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

        anno_norm = annos / 255.0
        n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * sigmoid(1 - anno_norm))
        p_energy = -np.log(anno_norm + EPSILON) / (tau * sigmoid(anno_norm))

        U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
        U[0, :] = n_energy.flatten()
        U[1, :] = p_energy.flatten()

        d.setUnaryEnergy(U)

        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

        # Do the inference
        infer = np.array(d.inference(1)).astype('float32')
        res = infer[1,:]

        # res *= 255 / res.max()
        res = res * 255
        res = res.reshape(img.shape[:2])
        cv2.imwrite(output, res.astype('uint8'))

        if count % count_num == count_num - 1:
            print((str(round(int(count + 1) / total_num * 100))) + '.0 %   ', end='')
