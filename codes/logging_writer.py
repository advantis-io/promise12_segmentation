import logging

import keras

class LoggingWriter(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        logging.debug("Running Batch: " + format(logs))

    def on_epoch_begin(self, epoch, logs=None):
        logging.info("Starting Epoch: {}".format(int(epoch)))

    def on_epoch_end(self, epoch, logs=None):
        logging.info("Summary for Epoch: {}".format(logs))

    def on_train_end(self, logs=None):
        pass