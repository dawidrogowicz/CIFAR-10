import numpy as np
import pickle


def number_to_one_hot(number):
    output = np.zeros(10)
    output[number] = 1
    return output


def prepare_files(file_path_format):
    data_x = []
    data_y = []

    for i in range(1, 6):
        file_path = file_path_format.format(i)
        with open(file_path, 'rb') as f:
            dict = pickle.load(f, encoding='bytes')

        x = dict[b'data']
        y = np.array(list(map(number_to_one_hot, dict[b'labels'])))

        if i < 2:
            data_x = x
            data_y = y
        else:
            data_x = np.concatenate((data_x, x))
            data_y = np.concatenate((data_y, y))

    data_x = np.reshape(data_x, (-1, 3, 1024))
    data_x = np.transpose(data_x, (0, 2, 1))
    data_x = np.reshape(data_x, (-1, 32, 32, 3))

    return data_x, data_y
