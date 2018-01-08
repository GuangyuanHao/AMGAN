from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
import copy
import os
from scipy.io import loadmat as loadimage
import numpy as np
import scipy
from skimage import feature
from PIL import Image
import cv2


def load_data(array):
    n =array.shape[0]
    size = array[0][0].shape
    imgA = array[0][0].reshape(1,size[0],size[1],size[2])
    for i in range(n-1):
        imgA = np.concatenate((imgA,array[i+1][0].reshape(1,size[0],size[1],size[2])),axis=0)
    return imgA/127.5-1.0
def rgb2gray(rgb):
    return np.dot(rgb[...,:3],[0.299, 0.587, 0.144])

def load_label(array):
    n =array.shape[0]
    hot_code = np.zeros(10).reshape(1,10)
    hot_code[0][array[0][1]]=1
    labelA = hot_code
    for i in range(n-1):
        hot_code = np.zeros(10).reshape(1, 10)
        hot_code[0][array[i+1][1]] = 1
        labelA = np.concatenate((labelA,hot_code),axis=0)
    return labelA

# ____________________________________________________

def save_images(image, size, path):
    return imsave(inverse_transform(image), size, path)

def imsave(image, size, path):
    return scipy.misc.imsave(path, merge(image, size), format='png')

def merge(image, size):
    [n, h, w, c] = image.shape
    image = image.reshape(n * h, w, c).astype(np.float)
    if c == 1:
        image = image.reshape(n * h, w)
    img = image[:h * size[0]]
    for i in range(size[1] - 1):
        img = np.concatenate((img, image[(i + 1) * h * size[0]:(i + 2) * h * size[0]]), axis=1)
    return img

def inverse_transform(image):
    return (image+1.)/2.