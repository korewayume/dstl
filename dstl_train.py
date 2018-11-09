# -*- coding: utf-8 -*-
import os
from glob import glob
from dstl.data_util import data_store
from dstl.data_gen import DataGen
from dstl.unet import unet
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint


def split_train_test_image_ids(test_size=0.2, random_state=20181109):
    path_pattern = os.path.join(data_store.three_band, "*.tif")
    image_ids = []
    for filename in glob(path_pattern):
        image_id = os.path.splitext(os.path.basename(filename))[0]
        image_ids.append(image_id)
    return train_test_split(sorted(image_ids), test_size=test_size, random_state=random_state, shuffle=True)


if __name__ == '__main__':
    image_size = (256, 256)
    batch_size = 32
    train, test = split_train_test_image_ids()
    train_gen = DataGen(train, "C", image_size, batch_size)
    test_gen = DataGen(test, "C", image_size, batch_size)
    model = unet('unet_checkpoint.hdf5', input_size=(*image_size, 3))
    model_checkpoint = ModelCheckpoint('unet_checkpoint.hdf5', monitor='loss', verbose=1, save_best_only=True)
    model.fit_generator(
        train_gen,
        validation_data=test_gen,
        callbacks=[model_checkpoint],
        steps_per_epoch=len(train_gen) // batch_size,
        epochs=10,
        validation_steps=len(test_gen) // batch_size,
        use_multiprocessing=False,
    )
