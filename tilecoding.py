
import h5py
from math import sqrt
import numpy as np

SAVER_WEIGHTS_KEY = "tile_coding_tiles"


class MultiTileCoder:
    def __init__(self, dims, limits, tilings, output_number, step_size=0.1, offset=lambda n: 2 * np.arange(n) + 1):
        self._output_numbers = output_number
        new_dim = dims.copy()
        new_dim.append(1)
        new_limits = limits.copy()
        new_limits.append((0, output_number))
        self.tile_coder = TileCoder(new_dim, new_limits, tilings, step_size, offset)

    def predict(self, input_data):
        predictions = np.zeros((input_data.shape[0], self._output_numbers))
        for i in range(input_data.shape[0]):
            single_prediction = []
            for action in range(self._output_numbers):
                single_prediction.append(self.tile_coder[np.append(input_data[i], [action])])
            predictions[i] = single_prediction
        return predictions

    def train_on_batch(self, Xs, Ys):
        total_loss = 0
        for i in range(Xs.shape[0]):
            for action in range(self._output_numbers):
                total_loss += sqrt((Ys[i][action] - self.tile_coder[np.append(Xs[i], [action])]) ** 2)
                self.tile_coder[np.append(Xs[i], [action])] = Ys[i][action]
        return total_loss

    def get_weights(self):
        return self

    def set_weights(self, weights):
        self.tile_coder.tiles = weights.tile_coder.tiles.copy()

    def save_weights(self, file_path):
        h5_file = h5py.File(file_path, "w")
        h5_file.create_dataset(SAVER_WEIGHTS_KEY, data=self.tile_coder.tiles)
        h5_file.close()

    def load_weights(self, file_path):
        h5_file = h5py.File(file_path, "r")
        self.tile_coder.tiles = h5_file[SAVER_WEIGHTS_KEY][:]
        h5_file.close()


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