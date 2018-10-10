#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 17:18:38 2017

@author: Inom Mirzaev / Martin Stypinski
"""

from __future__ import division, print_function

import logging
from functools import partial

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from skimage.exposure import equalize_adapthist
from sklearn.model_selection import KFold

from augmenters import *
from logging_writer import LoggingWriter
from metrics import dice_coef, dice_coef_loss, rel_abs_vol_diff
from metrics_callback import MetricsCallback
from model_mgpu import ModelMGPU
from models import *
from print_graph import plot_learning_performance
from sequencer import Sequencer


def start_logging():
    logging.basicConfig(
        format="%(asctime)s [%(threadName)-12.12s] [%(filename)-15.15s] [%(levelname)-8s]  %(message)s",
        handlers=[
            logging.FileHandler("app.log"),
            logging.StreamHandler()
        ],
        level=logging.INFO
    )


def img_resize(imgs, img_rows, img_cols, equalize=True):
    new_imgs = np.zeros([len(imgs), img_rows, img_cols])
    for mm, img in enumerate(imgs):
        if equalize:
            img = equalize_adapthist(img, clip_limit=0.05)
            # img = clahe.apply(cv2.convertScaleAbs(img))

        new_imgs[mm] = cv2.resize(img, (img_rows, img_cols), interpolation=cv2.INTER_NEAREST)

    return new_imgs


def load_data(img_rows, img_cols, train_list, val_list):
    logging.info("Training set: {}".format(train_list))
    logging.info("Test set: {}".format(val_list))

    fileList = os.listdir('../data/train/')
    fileList = sorted(filter(lambda x: '.mhd' in x, fileList))

    # val_list = [5, 15, 25, 35, 45]
    # train_list = list(set(range(50)) - set(val_list))
    count = 0
    for the_list in [train_list, val_list]:
        images = []
        masks = []

        filtered = filter(lambda x: any(str(ff).zfill(2) in x for ff in the_list), fileList)

        for filename in filtered:
            logging.info("Working on {}".format(filename))

            itkimage = sitk.ReadImage('../data/train/' + filename)
            imgs = sitk.GetArrayFromImage(itkimage)

            if 'segm' in filename.lower():
                imgs = img_resize(imgs, img_rows, img_cols, equalize=False)
                masks.append(imgs)

            else:
                imgs = img_resize(imgs, img_rows, img_cols, equalize=True)
                images.append(imgs)

        images = np.concatenate(images, axis=0).reshape(-1, img_rows, img_cols, 1)
        masks = np.concatenate(masks, axis=0).reshape(-1, img_rows, img_cols, 1)
        masks = masks.astype(int)

        # Smooth images using CurvatureFlow
        images = smooth_images(images)

        if count == 0:
            mu = np.mean(images)
            sigma = np.std(images)
            images = (images - mu) / sigma

            X_train = images
            y_train = masks
        elif count == 1:
            images = (images - mu) / sigma

            X_val = images
            y_val = images
        count += 1

    return X_train, y_train, X_val, y_val


def keras_fit_fold(fold_nr, train_index, test_index, img_rows=96, img_cols=96, n_imgs=10 ** 4, batch_size=32, workers=1):
    X_train_raw, y_train_raw, X_val, y_val = load_data(img_rows, img_cols, train_index, test_index)

    img_rows = X_train_raw.shape[1]
    img_cols = X_train_raw.shape[2]

    x, y = np.meshgrid(np.arange(img_rows), np.arange(img_cols), indexing='ij')
    elastic = partial(elastic_transform, x=x, y=y, alpha=img_rows * 1.5, sigma=img_rows * 0.07)
    # we create two instances with the same arguments
    data_gen_args = dict(
        featurewise_center=False,
        featurewise_std_normalization=False,
        rotation_range=10.,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=[1, 1.2],
        fill_mode='constant',
        preprocessing_function=elastic)

    training_sequence = Sequencer(X_train_raw, y_train_raw, sequence_size=n_imgs, batch_size=batch_size,
                                  data_gen_args=data_gen_args)

    raw_model = UNet((img_rows, img_cols, 1), start_ch=8, depth=7, batchnorm=True, dropout=0.5, maxpool=True,
                     residual=True)

    model = ModelMGPU(raw_model, 2)

    model.summary(print_fn=logging.info)
    model_checkpoint = ModelCheckpoint(
        '../data/weights-' + str(fold_nr) + '.h5', monitor='val_loss', save_best_only=True)

    c_backs = [model_checkpoint]
    c_backs.append(LoggingWriter())
    c_backs.append(MetricsCallback(X_train_raw, y_train_raw, X_val, y_val))

    model.compile(optimizer=Adam(lr=0.001), loss=dice_coef_loss, metrics=[dice_coef, rel_abs_vol_diff])

    history = model.fit_generator(
        training_sequence,
        epochs=10,
        verbose=1,
        shuffle=True,
        validation_data=(X_val, y_val),
        callbacks=c_backs,
        workers=workers,
        use_multiprocessing=True)

    logging.info(history.history)
    plot_learning_performance(history, 'plot-' + str(fold_nr) + '.png')


def keras_fit_generator(img_rows=96, img_cols=96, n_imgs=10 ** 4, batch_size=32, workers=1):
    # preprocess_data()

    kf = KFold(n_splits=5, shuffle=True)
    for fold_nr, (train_index, test_index) in enumerate(kf.split(range(0, 50))):
        logging.info("Starting Fold: {}".format(fold_nr))
        keras_fit_fold(fold_nr, train_index, test_index, img_rows, img_cols, n_imgs, batch_size, workers)




if __name__ == '__main__':
    import time
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

    start_logging()
    start = time.time()
    keras_fit_generator(img_rows=256, img_cols=256,
                        n_imgs=1000, batch_size=128, workers=16)

    # 15 * 10 ** 4

    end = time.time()

    logging.info('Elapsed time: {}'.format(round((end - start) / 60, 2)))
