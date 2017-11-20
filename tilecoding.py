#!/usr/bin/env python3

import h5py
from math import sqrt
import numpy as np

SAVER_WEIGHTS_KEY = "tile_coding_tiles"


class MultiTileCoder:
    def __init__(self, dims, limits, tilings, output_number, step_size=0.1, offset=lambda n: 2 * np.arange(n) + 1):
        self._output_numbers = output_number
        self._tile_coders = []
        for i in range(output_number):
            self._tile_coders.append(TileCoder(dims, limits, tilings, step_size, offset))

    def predict(self, input_data):
        predictions = np.zeros((input_data.shape[0], self._output_numbers))
        for i in range(input_data.shape[0]):
            single_prediction = []
            for t in self.tile_coders:
                single_prediction.append(t[input_data[i]])
            predictions[i] = single_prediction
        return predictions

    def train_on_batch(self, Xs, Ys):
        total_loss = 0
        for i in range(Xs.shape[0]):
            for j in range(self._output_numbers):
                total_loss += sqrt((Ys[i][j] - self._tile_coders[j][Xs[i]]) ** 2)
                self._tile_coders[j][Xs[i]] = Ys[i][j]
        return total_loss

    def get_weights(self) -> list:
        weights_list = []
        for t in self._tile_coders:
            weights_list.append(t.tiles.copy())
        return weights_list

    def set_weights(self, weights: list):
        # assert len(self._tile_coders) == len(weights)
        for i in range(len(weights)):
            self._tile_coders[i].tiles = weights[i]

    def save_weights(self, file_path):
        h5_file = h5py.File(file_path, "w")
        for i in range(len(self._tile_coders)):
            h5_file.create_dataset(SAVER_WEIGHTS_KEY + str(i), data=self._tile_coders[i].tiles)
        h5_file.close()

    def load_weights(self, file_path):
        h5_file = h5py.File(file_path, "r")
        current_index = 0
        while SAVER_WEIGHTS_KEY + str(current_index) in h5_file:
            self._tile_coders[current_index].tiles = h5_file[SAVER_WEIGHTS_KEY + str(current_index)][:]
            current_index += 1
        h5_file.close()

    @property
    def tile_coders(self):
        return self._tile_coders


class TileCoder:
    def __init__(self, dims, limits, tilings, step_size=0.1, offset=lambda n: 2 * np.arange(n) + 1):
        offset_vec = offset(len(dims))
        tiling_dims = np.array(dims, dtype=np.int) + offset_vec
        self._offsets = offset_vec * np.repeat([np.arange(tilings)], len(dims), 0).T / float(tilings)
        self._limits = np.array(limits)
        self._norm_dims = np.array(dims) / (self._limits[:, 1] - self._limits[:, 0])
        self._alpha = step_size / tilings
        self.tiles = np.zeros(tilings * np.prod(tiling_dims))
        self._tile_base_ind = np.prod(tiling_dims) * np.arange(tilings)
        self._hash_vec = np.ones(len(dims), dtype=np.int)
        for i in range(len(dims) - 1):
            self._hash_vec[i + 1] = tiling_dims[i] * self._hash_vec[i]

    def _get_tiles(self, x):
        off_coordinates = ((x - self._limits[:, 0]) * self._norm_dims + self._offsets).astype(int)
        return self._tile_base_ind + np.dot(off_coordinates, self._hash_vec)

    def __getitem__(self, x):
        tile_ind = self._get_tiles(x)
        return np.sum(self.tiles[tile_ind])

    def __setitem__(self, x, val):
        tile_ind = self._get_tiles(x)
        self.tiles[tile_ind] += self._alpha * (val - np.sum(self.tiles[tile_ind]))


def example():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import time

    # tile coder dimensions, limits, tilings, step size, and offset vector
    dims = [8, 8]
    lims = [(0, 2.0 * np.pi)] * 2
    tilings = 8
    alpha = 0.1

    # create tile coder
    T = tilecoder(dims, lims, tilings, alpha)

    # target function with gaussian noise
    def target_ftn(x, y, noise=True):
        return np.sin(x) + np.cos(y) + noise * np.random.randn() * 0.1

    # randomly sample target function until convergence
    timer = time.time()
    batch_size = 100
    for iters in range(100):
        mse = 0.0
        for b in range(batch_size):
            xi = lims[0][0] + np.random.random() * (lims[0][1] - lims[0][0])
            yi = lims[1][0] + np.random.random() * (lims[1][1] - lims[1][0])
            zi = target_ftn(xi, yi)
            T[xi, yi] = zi
            mse += (T[xi, yi] - zi) ** 2
        mse /= batch_size
        print('samples:', (iters + 1) * batch_size, 'batch_mse:', mse)
    print('elapsed time:', time.time() - timer)

    # get learned function
    print('mapping function...')
    res = 200
    x = np.arange(lims[0][0], lims[0][1], (lims[0][1] - lims[0][0]) / res)
    y = np.arange(lims[1][0], lims[1][1], (lims[1][1] - lims[1][0]) / res)
    z = np.zeros([len(x), len(y)])
    for i in range(len(x)):
        for j in range(len(y)):
            z[i, j] = T[x[i], y[j]]

    # plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, z, cmap=plt.get_cmap('hot'))
    plt.show()


if __name__ == '__main__':
    example()
