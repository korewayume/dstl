# -*- coding: utf-8 -*-
import os
import gdal
from gdalconst import GA_ReadOnly
import numpy as np
import pandas as pd
from shapely.wkt import loads as wkt_loads
from shapely import affinity
import cv2


# def geometry_to_pixel(coordinates, raster_size, xymax):
#     x_max, y_max = xymax
#     H, W = raster_size
#     W1 = 1.0 * W * W / (W + 1)
#     H1 = 1.0 * H * H / (H + 1)
#     xf = W1 / x_max
#     yf = H1 / y_max
#     coordinates[:, 1] *= yf
#     coordinates[:, 0] *= xf
#     return np.round(coordinates).astype(np.int32)
#
#
# def convert_contours(contours, raster_size, xymax):
#     if contours is None:
#         return None
#     exterior_list = []
#     interior_list = []
#     for k in range(len(contours)):
#         polygon = contours[k]
#         polygon_exterior = np.array(list(polygon.exterior.coords))
#         converted_exterior = geometry_to_pixel(polygon_exterior, raster_size, xymax)
#         exterior_list.append(converted_exterior)
#         for interior in polygon.interiors:
#             polygon_interior = np.array(list(interior.coords))
#             converted_interior = geometry_to_pixel(polygon_interior, raster_size, xymax)
#             interior_list.append(converted_interior)
#     return exterior_list, interior_list


def draw_contours_mask(mask, contours, mask_value=1):
    if contours is None:
        return mask

    exterior_list = []
    interior_list = []
    for polygon in contours:
        polygon_exterior = np.round(list(polygon.exterior.coords)).astype(np.int32)
        exterior_list.append(polygon_exterior)
        for interior in polygon.interiors:
            polygon_interior = np.round(list(interior.coords)).astype(np.int32)
            interior_list.append(polygon_interior)

    cv2.fillPoly(mask, exterior_list, mask_value)
    cv2.fillPoly(mask, interior_list, 0)
    return mask


def calc_xy_fact(raster_size, xmax_ymin):
    x_max, y_max = xmax_ymin
    h, w = raster_size
    xfact = 1.0 * w * (w / (w + 1)) / x_max
    yfact = 1.0 * h * (h / (h + 1)) / y_max
    return xfact, yfact


def vector_to_raster(raster_size, xmax_ymin, multipolygon, mask_value=1):
    xfact, yfact = calc_xy_fact(raster_size, xmax_ymin)
    mask = np.zeros(raster_size, np.uint8)
    if multipolygon:
        contours = affinity.scale(multipolygon, xfact=xfact, yfact=yfact, origin=(0, 0, 0))
        return draw_contours_mask(mask, contours, mask_value)
    else:
        return mask


class DataStore(object):
    @property
    def input_dir(self):
        return os.path.join(os.path.dirname(__file__), 'input')

    @property
    def sixteen_band(self):
        return os.path.join(self.input_dir, 'sixteen_band')

    @property
    def three_band(self):
        return os.path.join(self.input_dir, 'three_band')

    def __init__(self):
        train_wkt_csv = os.path.join(self.input_dir, 'train_wkt_v4.csv')
        grid_sizes_csv = os.path.join(self.input_dir, 'grid_sizes.csv')
        self.train_wkt_csv = pd.read_csv(train_wkt_csv)
        self.grid_sizes_csv = pd.read_csv(grid_sizes_csv, names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)

    def get_multipolygon_by_image_class(self, image_id, klass):
        image = self.train_wkt_csv[self.train_wkt_csv.ImageId == image_id]
        multipolygon_wkt = image[image.ClassType == klass].MultipolygonWKT
        multipolygon = None
        if len(multipolygon_wkt) > 0:
            assert len(multipolygon_wkt) == 1
            multipolygon = wkt_loads(multipolygon_wkt.values[0])
        return multipolygon

    def get_image_xmax_ymin(self, image_id):
        return self.grid_sizes_csv[self.grid_sizes_csv.ImageId == image_id].iloc[0, 1:].astype(float)

    def raster_filename(self, image_id, raster_type):
        types = {
            "A": os.path.join(self.sixteen_band, '{}_A.tif'),
            "M": os.path.join(self.sixteen_band, '{}_M.tif'),
            "P": os.path.join(self.sixteen_band, '{}_P.tif'),
            "C": os.path.join(self.three_band, '{}.tif'),
        }
        return types[raster_type].format(image_id)

    def raster_image(self, image_id, raster_type):
        filename = self.raster_filename(image_id, raster_type)
        raster = gdal.Open(filename, GA_ReadOnly)
        image = raster.ReadAsArray().transpose(1, 2, 0)
        return image

    def get_raster_size(self, image_id, raster_type):
        dataset = gdal.Open(self.raster_filename(image_id, raster_type), GA_ReadOnly)
        return dataset.RasterYSize, dataset.RasterXSize

    def mask_for_image_and_class(self, image_id, raster_type, classes):
        xmax_ymin = self.get_image_xmax_ymin(image_id)
        raster_size = self.get_raster_size(image_id, raster_type)
        masks = []
        for klass in classes:
            mask = np.zeros(raster_size, np.float32)
            multipolygon = self.get_multipolygon_by_image_class(image_id, klass)
            mask_klass = vector_to_raster(raster_size, xmax_ymin, multipolygon, klass)
            mask[mask_klass != 0] = 1.0
            masks.append(mask)
        return np.stack(masks, axis=-1)


data_store = DataStore()
