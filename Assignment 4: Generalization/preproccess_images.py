import os
import pickle
import random
from PIL import Image
import numpy as np


def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def get_image_from_vector(vec):
    rgb = [vec[:1024], vec[1024:2048], vec[2048:]]
    img_arr = []
    for color in rgb:
        color_arr = []
        for start_ind in range(0, 1024, 32):
            row = color[start_ind:start_ind+32]
            color_arr.append(row)
        img_arr.append(color_arr)
    img_arr = np.array(img_arr)
    img_arr = np.transpose(img_arr)
    return Image.fromarray(img_arr)


batch_file = './cifar-10-batches-py/data_batch_{}'
meta_file = './cifar-10-batches-py/batches.meta'
meta_dict = unpickle(meta_file)
label_names  = meta_dict[b"label_names"]
label_names = [name.decode() for name in label_names]

to_print = [f"'{j}':{i}" for i, j in enumerate(label_names)]
print("{", f"{','.join(to_print)}", "}")

img_path_format = "./inception/data/{label_name}/{image_name}.jpg"
dir_path_format = "./inception/data/{label_name}"

# for bach_number in range(1,6):
#     file_name = batch_file.format(bach_number)
#     batch = unpickle(file_name)
#     data = batch[b"data"]
#     labels = batch[b"labels"]
#
#     for i in range(len(data)):
#         img = get_image_from_vector(data[i])
#         label_number = labels[i]
#         label_name = label_names[label_number]
#
#         dir_path = dir_path_format.format(label_name=label_name)
#         if not os.path.isdir(dir_path):
#             os.mkdir(dir_path)
#
#         img_path = img_path_format.format(label_name=label_name, image_name=i)
#         img.save(img_path)
