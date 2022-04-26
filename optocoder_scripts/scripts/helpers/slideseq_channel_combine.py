import numpy as np
import pandas as pd
import tifffile as tf
import os
import tifftools

def only_numerics(seq):
    """This is a helper to get the digits from the file name"""
    seq_type = type(seq)
    return seq_type().join(filter(seq_type.isdigit, seq))

folder_path = "/data/rajewsky/home/asenel/projects/deployment/optocoder/datasets/imaging/slideseq/v1_180430_1/raw"
image_paths = []
image_names = []
for file in os.listdir(folder_path):
        if file.endswith(".tif"):
            image_names.append(file)

image_names = sorted(image_names, key=lambda x:int(only_numerics(x.split('_')[4])))
for image in image_names:
    image_paths.append(os.path.join(folder_path, image))
image_paths = np.array(image_paths).reshape(20,4)

for i, image_path in enumerate(image_paths):
    print(image_path)
    input1 = tifftools.read_tiff(image_path[0])
    input2 = tifftools.read_tiff(image_path[1])
    input3 = tifftools.read_tiff(image_path[2])
    input4 = tifftools.read_tiff(image_path[3])

    input1['ifds'].extend(input2['ifds'])
    input1['ifds'].extend(input3['ifds'])
    input1['ifds'].extend(input4['ifds'])

    tifftools.write_tiff(input1, os.path.join('/data/rajewsky/home/asenel/projects/deployment/optocoder/datasets/imaging/slideseq/v1_180430_1', f'v1_180430_1_Ligation_{i}.tif'))