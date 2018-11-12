# -*- coding: utf-8 -*-
import os
import zipfile

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

os.makedirs("three_band", exist_ok=True)

with zipfile.ZipFile('three_band.zip') as zf:
    for image_id in image_ids:
        filename = 'three_band/{}.tif'.format(image_id)
        data = zf.read(filename)
        with open(filename, "wb+") as f:
            f.write(data)
