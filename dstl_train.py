# -*- coding: utf-8 -*-
import os
import datetime
from glob import glob
from dstl.data_util import data_store
from dstl.data_gen import DataGen
from dstl.unet import unet
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, TensorBoard

"""
01. Buildings - large building, residential, non-residential, fuel storage facility, fortified building
02. Misc. Manmade structures 
03. Road 
04. Track - poor/dirt/cart track, footpath/trail
05. Trees - woodland, hedgerows, groups of trees, standalone trees
06. Crops - contour ploughing/cropland, grain (wheat) crops, row (potatoes, turnips) crops
07. Waterway 
08. Standing water
09. Vehicle Large - large vehicle (e.g. lorry, truck,bus), logistics vehicle
10. Vehicle Small - small vehicle (car, van), motorbike
"""


def split_train_test_image_ids(test_size=0.2, random_state=20181109):
    # path_pattern = os.path.join(data_store.three_band, "*.tif")
    # image_ids = []
    # for filename in glob(path_pattern):
    #     image_id = os.path.splitext(os.path.basename(filename))[0]
    #     image_ids.append(image_id)

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


if __name__ == '__main__':
    image_size = (256, 256)
    batch_size = 25
    classes = [1, 2, 3, 4, 5]
    train, test = split_train_test_image_ids()
    train_gen = DataGen(train, "C", image_size, batch_size, classes=classes)
    test_gen = DataGen(test, "C", image_size, batch_size, classes=classes)
    if os.path.exists('unet_checkpoint.hdf5'):
        model = unet('unet_checkpoint.hdf5', input_size=(*image_size, 3), num_classes=len(classes))
    else:
        model = unet(input_size=(*image_size, 3), num_classes=len(classes))
    model_checkpoint = ModelCheckpoint('unet_checkpoint.hdf5', monitor='loss', verbose=1, save_best_only=True)
    tensor_board = TensorBoard(log_dir='{:%m-%d_%H:%M_logs_%S}'.format(datetime.datetime.now()))
    model.fit_generator(
        train_gen,
        validation_data=test_gen,
        callbacks=[model_checkpoint, tensor_board],
        steps_per_epoch=len(train_gen),
        epochs=10,
        validation_steps=len(test_gen),
        use_multiprocessing=False,
    )
