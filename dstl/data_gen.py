# -*- coding: utf-8 -*-
from itertools import product
import numpy as np
from keras.preprocessing.image import Iterator
from .data_util import data_store
from .preprocessing import truncate_as_float32, clip_to_tile


class DataGen(Iterator):

    def __init__(self, image_ids, image_type, target_size, batch_size, classes=None, shuffle=True, seed=None):
        target_height, target_width = target_size
        tile_names = []
        for image_id in image_ids:
            height, width = data_store.get_raster_size(image_id, image_type)
            rows = int(np.ceil(height / target_height))
            cols = int(np.ceil(width / target_width))
            for r, c in product(range(rows), range(cols)):
                tile_names.append((image_id, r, c))

        super().__init__(len(tile_names), batch_size, shuffle, seed)
        self.tile_names = np.array(tile_names)
        self.image_type = image_type
        self.target_size = target_size
        if classes is None:
            self.classes = list(range(1, 11))
        else:
            self.classes = classes

    def _get_batches_of_transformed_samples(self, index_array):
        tile_names = self.tile_names[index_array].tolist()
        x_data = []
        y_data = []
        for image_id, r, c in tile_names:
            r = int(r)
            c = int(c)
            x = clip_to_tile(truncate_as_float32(
                data_store.raster_image(image_id, self.image_type)
            ), self.target_size)[r, c]
            y = clip_to_tile(
                data_store.mask_for_image_and_class(image_id, self.image_type, self.classes),
                self.target_size
            )[r, c]
            x_data.append(x)
            y_data.append(y)

        return np.stack(x_data), np.stack(y_data)
