import logging

import numpy as np
from keras import callbacks

from metrics import surface_dist
from test import resize_pred_to_val, numpy_dice

import matplotlib.pyplot as plt

logging.getLogger("matplotlib").setLevel(logging.CRITICAL)


class MetricsCallback(callbacks.Callback):
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_trian = y_train
        self.y_test = y_test
        self.test_loss = []
        self.train_loss = []
        self.test_dice = []
        self.train_dice = []

        self.mean_dice = []
        self.std_dice = []
        self.mean_hausdorff = []
        self.std_hausdorff = []

    def on_epoch_end(self, batch, logs={}):
        y_pred = self.model.predict(self.X_train, verbose=1, batch_size=128)
        print('Results on train set:')
        print('Dice Accuracy:', numpy_dice(self.y_train, y_pred))

        y_pred = self.model.predict(self.X_test, verbose=1, batch_size=128)
        print('Results on validation set')
        print('Accuracy:', numpy_dice(self.y_test, y_pred))

        vol_scores = []
        hauss_dist = []

        end_ind = 0
        for data_obj in self.train_set:
            y_true = data_obj.mask
            spacing = data_obj.mask_spacing

            start_ind = end_ind
            end_ind += len(y_true)

            y_pred_up = resize_pred_to_val(y_pred[start_ind:end_ind], y_true.shape)

            vol_scores.append(numpy_dice(y_true, y_pred_up, axis=None))
            surfd = surface_dist(y_true, y_pred_up, sampling=spacing)
            hauss_dist.append(surfd.max())

        vol_scores = np.array(vol_scores)

        print('Mean volumetric DSC:', vol_scores.mean())
        print('Std volumetric DSC:', vol_scores.std())
        print('Mean Haussdorf. Dist:', np.mean(hauss_dist))
        print('Std Haussdorf DSC:', np.std(hauss_dist))

        self.mean_hausdorff.append(np.mean(hauss_dist))
        self.std_hausdorff.append(np.std(hauss_dist))
        self.mean_dice.append(vol_scores.mean())
        self.std_dice.append(vol_scores.std())

        return

    def on_train_end(self, logs=None):
        logging.info("Mean Dice: {}".format(self.mean_dice))
        logging.info("Std Dice: {}".format(self.std_dice))
        logging.info("Mean Haussdorf: {}".format(self.mean_hausdorff))
        logging.info("Std Hausdorf: {}".format(self.std_hausdorff))
        super().on_train_end(logs)

    def save(self, path):
        epochs = list(range(1, len(self.mean_dice) + 1))

        _, axes = plt.subplots(nrows=2, ncols=2, sharex=True)

        ax2 = axes[0]
        ax2.plot(epochs, history.history['loss'], label='Training')
        ax2.set_title('Loss')
        plt.xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.legend()

        ax4 = axes[1]
        ax4.set_ylim(ax2.get_ylim())
        ax4.plot(epochs, smooth_curve(history.history['loss']), label='Training')
        if with_validation:
            ax4.plot(epochs, smooth_curve(history.history['val_loss']), label='Validation')
        ax4.set_title('Loss EMA')
        plt.xlabel('Epochs')
        ax4.legend()

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(experiment_file)

        plt.clf()
        plt.cla()
        plt.close()
