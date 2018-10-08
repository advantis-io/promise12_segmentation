import random

from keras.utils import Sequence
import numpy as np
from keras_preprocessing.image import ImageDataGenerator


class Sequencer(Sequence):
    def __init__(self, X, y, batch_size, sequence_size, data_gen_args):
        """A `Sequence` implementation that can augment data
            X: The numpy array of inputs.
            y: The numpy array of targets.
            batch_size: The generator mini-batch size.
            sequence_size: The number of elements in the sequence
            data_gen_args: The arguments for the ImageDataGenerator to apply on X, y
        """

        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.sequence_encoding = random.sample(range(0, sequence_size), sequence_size)
        self.imgaug = ImageDataGenerator(data_gen_args)

    def __len__(self):
        return len(self.sequence_encoding) // self.batch_size

    def on_epoch_end(self):
        pass

    def __getitem__(self, batch_idx):
        batch_X = np.zeros((self.batch_size, self.X.shape[1], self.X.shape[2], self.X.shape[3]))
        batch_y = np.zeros((self.batch_size, self.y.shape[1], self.y.shape[2], self.y.shape[3]))

        for i in range(self.batch_size):
            cur_real_idx = (batch_idx + i) % self.X.shape[0]
            # This creates a dictionary with the params
            params = self.imgaug.get_random_transform(batch_X[i].shape,
                                                      seed=self.sequence_encoding[batch_idx * self.batch_size + i])
            # We can now deterministicly augment all the images
            batch_X[i] = self.imgaug.apply_transform(self.X[cur_real_idx], params)
            batch_y[i] = self.imgaug.apply_transform(self.y[cur_real_idx], params)

        return batch_X, batch_y
