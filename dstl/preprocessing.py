# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np


def truncate_as_float32(image, alpha=0.02):
    float32_image = np.zeros_like(image, dtype=np.float32)
    min_max = np.quantile(image, [alpha, 1 - alpha], axis=[0, 1], interpolation='nearest')
    for c in range(float32_image.shape[-1]):
        min_val, max_val = min_max[..., c]
        single_channel = image[..., c].astype(np.float32)
        single_channel[single_channel < min_val] = min_val
        single_channel[single_channel > max_val] = max_val
        float32_image[..., c] = (single_channel - min_val) / (max_val - min_val)
    return float32_image


def clip_to_tile(image, target_size):
    """将一张图片切成(target_height, target_width)大小的瓦片

    Parameters
    -----------
    image : ndarray
        图片数组
    target_size : tuple of integer
        瓦片尺寸(target_height, target_width)

    Returns
    -------
    cliped : ndarray
        形如(rows, cols, target_height, target_width)

    Examples
    --------
    >>> image = np.arange(25).reshape(5,5)
    >>> clip_to_tile(image, (3, 3))
    array([[[[ 0,  1,  2],
             [ 5,  6,  7],
             [10, 11, 12]]]])
    >>> clip_to_tile(image, (3, 3))
    array([[[[ 0,  1,  2],
             [ 5,  6,  7],
             [10, 11, 12]],

            [[ 3,  4,  0],
             [ 8,  9,  0],
             [13, 14,  0]]],


           [[[15, 16, 17],
             [20, 21, 22],
             [ 0,  0,  0]],

            [[18, 19,  0],
             [23, 24,  0],
             [ 0,  0,  0]]]])
    """
    target_height, target_width = target_size
    height, width = image.shape[:2]
    rows = int(np.ceil(height / target_height))
    cols = int(np.ceil(width / target_width))
    if len(image.shape) == 3:
        channels = image.shape[2]
        rv = np.zeros((rows, cols, target_height, target_width, channels), dtype=image.dtype)
    else:
        rv = np.zeros((rows, cols, target_height, target_width), dtype=image.dtype)
    for y in range(rows):
        for x in range(cols):
            y_s, y_e = y * target_height, min(y * target_height + target_height, height)
            x_s, x_e = x * target_width, min(x * target_width + target_width, width)
            rv[y, x][:y_e - y_s, :x_e - x_s] = image[y_s: y_e, x_s: x_e]
    return rv
