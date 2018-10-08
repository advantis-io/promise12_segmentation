import random

import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from keras_preprocessing.image import ImageDataGenerator


class DualImageGenerator:
    """
    This image data generator executes the data augmentation in parallel.
    """

    def __init__(self, **kwArgs):
        self.max_seed = 4294967296
        self.image_data_generator = ImageDataGenerator(**kwArgs)
        self.mask_data_generator = ImageDataGenerator(**kwArgs)

    def flow(self, **kwArgs):
        print(kwArgs['sequence'])
        if not ('image' in kwArgs.keys() or 'segment' in kwArgs.keys()):
            raise Exception('image and segment needs to be provided!')

        if not 'seed' in kwArgs.keys():
            kwArgs['seed'] = random.randint(0, self.max_seed)
            logging.warning('Seed value is beeing generated: {}'.format(kwArgs['seed']))

        image = kwArgs['image']
        segment = kwArgs['segment']

        kwArgs.pop('image', None)
        kwArgs.pop('segment', None)

        kwArgs['x'] = image
        iterator_img = self.image_data_generator.flow(**kwArgs)

        kwArgs['x'] = segment
        iterator_mask = self.mask_data_generator.flow(**kwArgs)
        return zip(iterator_img, iterator_mask)
