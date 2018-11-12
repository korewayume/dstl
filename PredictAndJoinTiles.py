# -*- coding: utf-8 -*-
import os
import numpy as np
from osgeo import gdal
from itertools import product
from dstl.unet import unet
from dstl.data_util import data_store
from dstl.preprocessing import truncate_as_float32, clip_to_tile
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def split_train_test_image_ids(test_size=0.2, random_state=20181109):
    image_ids = [
        '6070_2_3',
        '6060_2_3',
        '6100_1_3',
        '6110_1_2',
        '6140_3_1',
        '6100_2_2',
        '6100_2_3',
        '6120_2_2',
        '6140_1_2',
        '6120_2_0',
        '6110_3_1',
        '6110_4_0',
    ]
    return train_test_split(sorted(image_ids), test_size=test_size, random_state=random_state, shuffle=True)


def save_tif(array, filename):
    ysize, xsize, bands = array.shape
    driver = gdal.GetDriverByName('GTiff')
    raster = driver.Create(filename, xsize, ysize, bands, gdal.GDT_Float32, options=["COMPRESS=LZW"])
    for band in range(bands):
        raster_band = raster.GetRasterBand(band + 1)
        raster_band.WriteArray(array[..., band])
    raster.FlushCache()


def join_tiles(tiles):
    rows, cols, h, w = tiles.shape[:4]
    rv = np.zeros((rows * h, cols * w, *tiles.shape[4:]), dtype=tiles.dtype)
    for r, c in product(range(rows), range(cols)):
        y_s, y_e = r * h, r * h + h
        x_s, x_e = c * w, c * w + w
        rv[y_s: y_e, x_s: x_e] = tiles[r, c]
    return rv


def get_data(image_id, target_size=(256, 256)):
    x = clip_to_tile(truncate_as_float32(data_store.raster_image(image_id, 'C')), target_size)
    y = clip_to_tile(data_store.mask_for_image_and_class(image_id, 'C', [1, 2, 3, 4, 5]), target_size)
    return x, y


if __name__ == '__main__':
    model = unet('unet_checkpoint.hdf5', input_size=(256, 256, 3), num_classes=5)
    train, test = split_train_test_image_ids()
    os.makedirs("predict_result", exist_ok=True)
    for image_ids, train_test in ((train, "train"), (test, "test")):
        for image_id in tqdm(image_ids, desc=train_test):
            YSize, XSize = data_store.get_raster_size(image_id, 'C')
            x, y = get_data(image_id)
            tile_shape = list(x.shape)
            tile_shape[-1] = 5
            tile_shape = tuple(tile_shape)
            p = model.predict(x.reshape((-1, 256, 256, 3))).reshape(tile_shape)
            Y = join_tiles(y)[:YSize, :XSize]
            P = join_tiles(p)[:YSize, :XSize]
            save_tif(Y, "predict_result/{}_Y_{}.tif".format(image_id, train_test))
            save_tif(P, "predict_result/{}_P_{}.tif".format(image_id, train_test))
