"""import torch
import os
import numpy as np
import matplotlib.pyplot as plt


# basedir = "/data1slow/users/asuso/trans_bezier"
basedir = "/home/asuso/PycharmProjects/trans_bezier"

i = 0
num_samples = 0
for filename in os.listdir(basedir+"/Datasets/QuickDraw/npy_files"):
    if filename.endswith(".npy"):
        data_array = np.load(basedir+"/Datasets/QuickDraw/"+filename)
        num_samples += data_array.shape[0]

        plt.imshow(data_array[0].reshape(28, 28), cmap='gray')
        plt.title("{}".format(filename))
        plt.show()

print("El dataset sencer te mida", num_samples)


import struct
from struct import unpack
import pickle


def unpack_drawing(file_handle):
    key_id, = unpack('Q', file_handle.read(8))
    country_code, = unpack('2s', file_handle.read(2))
    recognized, = unpack('b', file_handle.read(1))
    timestamp, = unpack('I', file_handle.read(4))
    n_strokes, = unpack('H', file_handle.read(2))
    image_x = []
    image_y = []
    for i in range(n_strokes):
        n_points, = unpack('H', file_handle.read(2))
        fmt = str(n_points) + 'B'
        x = unpack(fmt, file_handle.read(n_points))
        y = unpack(fmt, file_handle.read(n_points))
        image_x.append(x)
        image_y.append(y)

    return (image_x, image_y)

def unpack_drawings(filename):
    with open(filename, 'rb') as f:
        while True:
            try:
                yield unpack_drawing(f)
            except struct.error:
                break

for filename in os.listdir(basedir+"/Datasets/QuickDraw/selected_bin"):
    image_list = []
    if filename.endswith(".bin"):
        i = 0
        for image in unpack_drawings(basedir+"/Datasets/QuickDraw/selected_bin/"+filename):
            i += 1
            im = torch.zeros((258, 258))
            image_x, image_y = image
            for stroke_x, stroke_y in zip(image_x, image_y):
                str_x = torch.tensor(stroke_x)
                str_y = torch.tensor(stroke_y)
                im[str_x+1, str_y+1] = 1
            image_list.append(im)
            if i >= 10000:
                break

        category_output = torch.stack(image_list, dim=0).unsqueeze(1)
        with open(basedir+"/Datasets/QuickDraw/processed_bin/"+filename+'.pickle', 'wb') as handle:
            pickle.dump(category_output, handle, protocol=pickle.HIGHEST_PROTOCOL)"""

import torch
import os
import pickle
import json
import matplotlib.pyplot as plt

# basedir = "/data1slow/users/asuso/trans_bezier"
basedir = "/home/asuso/PycharmProjects/trans_bezier"


for filename in os.listdir(basedir+"/Datasets/QuickDraw/selected_raw"):
    image_list = []
    with open(basedir+"/Datasets/QuickDraw/selected_raw/"+filename, "r") as file:
        images = []
        dist_x = 0
        dist_y = 0
        i = 0
        for doc in file:
            j_content = json.loads(doc)
            draw = j_content["drawing"]

            if j_content["recognized"]:
                x_list = []
                y_list = []
                strokes_len = []
                for stroke in draw:
                    x, y, t = stroke
                    x_list += x
                    y_list += y
                    strokes_len.append(len(x))

                dist_x = max(x_list)-min(x_list)
                dist_y = max(y_list)-min(y_list)

                x = torch.tensor(x_list, dtype=torch.float32) - min(x_list)
                y = torch.tensor(y_list, dtype=torch.float32) - min(y_list)

                d = max(dist_x, dist_y)
                red_factor = d/60
                x /= red_factor
                y /= red_factor
                x = x.long()
                y = y.long()

                image = torch.zeros(64, 64, dtype=torch.float32)
                points_processed = 0
                for stroke_len in strokes_len:
                    for j in range(1, stroke_len):
                        x_seq = torch.linspace(x[points_processed+j-1], x[points_processed+j], 10).long()
                        y_seq = torch.linspace(y[points_processed+j-1], y[points_processed+j], 10).long()
                        image[y_seq+2, x_seq+2] = 1
                    points_processed += stroke_len

                images.append(image)

                """plt.imshow(image, cmap='gray')
                plt.title("{}".format(filename))
                plt.show()"""

                if i == 19999:
                    break
                i += 1

        category_output = torch.stack(images, dim=0).unsqueeze(1)
        with open(basedir + "/Datasets/QuickDraw/processed_raw/" + filename + '.pickle', 'wb') as handle:
            pickle.dump(category_output, handle, protocol=pickle.HIGHEST_PROTOCOL)



    """for image in unpack_drawings(basedir+"/Datasets/QuickDraw/selected_bin/"+filename):
        i += 1
        im = torch.zeros((258, 258))
        image_x, image_y = image
        for stroke_x, stroke_y in zip(image_x, image_y):
            str_x = torch.tensor(stroke_x)
            str_y = torch.tensor(stroke_y)
            im[str_x+1, str_y+1] = 1
        image_list.append(im)
        if i >= 10000:
            break

    category_output = torch.stack(image_list, dim=0).unsqueeze(1)
    with open(basedir+"/Datasets/QuickDraw/processed_bin/"+filename+'.pickle', 'wb') as handle:
        pickle.dump(category_output, handle, protocol=pickle.HIGHEST_PROTOCOL)


for filename in os.listdir(basedir+"/Datasets/QuickDraw/processed_bin"):
    images = pickle.load(open(basedir+"/Datasets/QuickDraw/processed_bin/"+filename, 'rb'))
    for i in range(10):
        plt.imshow(images[i, 0], cmap='gray')
        plt.title("{}".format(filename))
        plt.show()"""
