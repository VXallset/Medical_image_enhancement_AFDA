from skimage import filters, io
import numpy as np
import cv2
from scipy import signal
import utils
from matplotlib import pyplot as plt


def get_obj_mask(image):
    """
    Get the object in an image.
    :param raw_image:
    :return: mask of ROI of the object
    """
    float_image = np.float32(image)
    otsu_threshold = filters.threshold_otsu(float_image)
    otsu_mask = float_image < otsu_threshold
    return otsu_mask


def get_ROI(raw_image):
    """
    Get the ROI of the input raw image.
    :param raw_image: grayscale image.
    :return: grayscale image.
    """

    float_image = np.float32(raw_image)
    shape_height, shape_width = raw_image.shape
    otsu_threshold = filters.threshold_otsu(float_image)
    otsu_mask = float_image < otsu_threshold
    int_mask = np.ones_like(float_image) * otsu_mask
    kernel = np.ones((5, 5), np.int64)
    gradient = cv2.morphologyEx(int_mask, cv2.MORPH_GRADIENT, kernel)
    gradient_mask = gradient > 0

    coordinate_w = np.array([[y for y in range(shape_width)] for _ in range(shape_height)], dtype=np.int64)
    coordinate_h = np.array([[x for _ in range(shape_width)] for x in range(shape_height)], dtype=np.int64)
    coordinate_w = coordinate_w[gradient_mask]
    coordinate_h = coordinate_h[gradient_mask]

    min_h, min_w = np.min(coordinate_h), np.min(coordinate_w)
    max_h, max_w = np.max(coordinate_h), np.max(coordinate_w)
    result = raw_image[min_h:max_h, min_w:max_w]
    return result


def construct_pdm(v):
    """
    Construct the partial differential kernel for a given v.
    :param v: np.float
    :return: The constructed kernel.
    """
    kernel = np.zeros((5, 5))
    kernel[0::2, 0::2] = (v * v -v) / 2.0
    kernel[1:-1, 1:-1] = -v
    kernel[2, 2] = 8.0
    kernel = kernel /(8.0 - 12.0 * v + 4 * v * v)
    return kernel
#print(np.sum(construct_pdm(0.5)))


def get_mean_gradient(image):
    """
    Calculate the mean gradient of an image. Using zero padding.
    :param image: the input image. M x N array
    :return: the mean gradiant of an image. M x N array
    """
    kernel = -np.ones((3, 3), dtype=np.float32)
    kernel[1, 1] = 8.0
    #print(kernel)

    #result = signal.convolve2d(image, kernel)
    result = cv2.filter2D(image, -1, kernel/8.0, borderType=0)
    """
    shape_1, shape_2 = image.shape
    pad_image = np.zeros((shape_1 + 2, shape_2 + 2))
    pad_image[1:-1, 1:-1] = image
    result = np.zeros_like(image)
    for i in range(1, shape_1):
        for j in range(1, shape_2):
            result[i, j] = np.abs(np.sum(kernel * pad_image[i-1:i+2, j-1:j+2])) / 8.0
    """
    return np.abs(result)


def convolution_fix_order(image, v):
    """
    Main part of AFDA.
    :param image: ROI np.int64 nd array
    :return:
    """
    #image = util.set_min_max_window_with_gamma_correction(image, 2.2)
    #image = util.deal_with_out_boundary(image)
    background_mask = image == np.min(image)
    mask = construct_pdm(v)
    filted_image = cv2.filter2D(image, -1, mask)
    foreground_mask = ~ background_mask

    foreground_min = np.min(filted_image[foreground_mask])
    foreground_max = np.max(filted_image[foreground_mask])

    result = (filted_image - foreground_min) / (foreground_max - foreground_min)
    result[background_mask] = 0
    return result

def convolution_adaptive_order(image):
    """
    Main part of AFDA.
    :param image: ROI np.int64 nd array
    :return:
    """
    #image = util.set_min_max_window_with_gamma_correction(image, 2.2)
    #image = util.deal_with_out_boundary(image)

    roi_obj_mask = get_obj_mask(image)

    mean_gradient = get_mean_gradient(image)
    mean_gradient[0, :] = 0.0
    mean_gradient[-1, :] = 0.0
    mean_gradient[:, 0] = 0.0
    mean_gradient[:, -1] = 0.0

    shape_width, shape_height = image.shape
    pad_image = np.zeros((shape_width + 5, shape_height + 5))
    pad_image[2:-3, 2:-3] = image

    filted_image = np.zeros_like(image)

    obj_mean_gradient = mean_gradient[roi_obj_mask]

    otsu_threshold = filters.threshold_otsu(obj_mean_gradient)
    print('otsu_threshold = {}'.format(otsu_threshold))

    Q = np.mean(obj_mean_gradient)
    Med = np.mean(obj_mean_gradient[obj_mean_gradient >= otsu_threshold])
    Mtex = np.mean(obj_mean_gradient[obj_mean_gradient < otsu_threshold])
    v1 = (Med - Q) / Med
    v2 = (Q - Mtex) / Q
    #v1 = 0.66
    #v2 = 0.18
    print('Q = {}, Med = {}, Mtex = {}'.format(Q, Med, Mtex))
    print('v1 = {}, v2 = {}'.format(v1, v2))

    for i in range(shape_width):
        for j in range(shape_height):
            v = 0

            if not roi_obj_mask[i,j]:
                continue

            Mij = mean_gradient[i, j]
            if Mij >= otsu_threshold:
                if (Mij - otsu_threshold) / Mij >= v1:
                    v = (Mij - otsu_threshold) / Mij
                    #print('(Mij - otsu_threshold) / Mij = {}'.format(v))
                else:
                    v = v1
                    #print('v1 = {}'.format(v))
            else:
                if Mij > 2:
                    if Mij / otsu_threshold >= v2:
                        v = v2
                        #print('v2 = {}'.format(v2))
                    else:
                        v = Mij / otsu_threshold
                        #print('Mij / otsu_threshold = {}'.format(v))
                else:
                    v = 0
                    #print('v=0')

            kernel = construct_pdm(v)
            filted_image[i, j] = np.sum(kernel * pad_image[i: i+5, j:j+5])
    result = np.uint8(255 * (filted_image - np.min(filted_image)) / (np.max(filted_image) - np.min(filted_image)))
    return result
