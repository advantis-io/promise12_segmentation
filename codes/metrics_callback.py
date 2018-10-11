from keras import callbacks

from metrics import rel_abs_vol_diff, surface_dist
from test import resize_pred_to_val, numpy_dice, read_cases
import numpy as np

from train import extract_data


class MetricsCallback(callbacks.Callback):
    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set

    def on_train_begin(self, logs={}):
        self.X_train
        self.y_train
        #
        # #X_train, y_train = extract_data(self.train_set)
        #
        # X_train = np.concatenate(X_train, axis=0).reshape(-1, img_rows, img_cols, 1)
        # y_train = np.concatenate(y_train, axis=0).reshape(-1, img_rows, img_cols, 1)
        # y_train = y_train.astype(int)
        #
        # # Smooth images using CurvatureFlow
        # #X_train = smooth_images(X_train)
        #
        # mu = np.mean(X_train)
        # sigma = np.std(X_train)
        # X_train = (X_train - mu) / sigma
        #
        # X_test, y_test = extract_data(train_set)
        #
        # X_test = np.concatenate(X_test, axis=0).reshape(-1, img_rows, img_cols, 1)
        # y_test = np.concatenate(y_test, axis=0).reshape(-1, img_rows, img_cols, 1)
        # y_test = y_test.astype(int)
        #
        # X_test = smooth_images(X_test)
        # X_test = (X_test - mu) / sigma
        #
        # self._data = []

    def on_epoch_end(self, batch, logs={}):
        y_pred = self.model.predict(self.X_train, verbose=1, batch_size=128)
        print('Results on train set:')
        print('Accuracy:', numpy_dice(self.y_train, y_pred))

        y_pred = self.model.predict(self.X_test, verbose=1, batch_size=128)
        print('Results on validation set')
        print('Accuracy:', numpy_dice(self.y_test, y_pred))

        # vol_scores = []
        # ravd = []
        # scores = []
        # hauss_dist = []
        # mean_surf_dist = []
        #
        # start_ind = 0
        # end_ind = 0
        # for y_true, spacing in read_cases():
        #     start_ind = end_ind
        #     end_ind += len(y_true)
        #
        #     y_pred_up = resize_pred_to_val(y_pred[start_ind:end_ind], y_true.shape)
        #
        #     ravd.append(rel_abs_vol_diff(y_true, y_pred_up))
        #     vol_scores.append(numpy_dice(y_true, y_pred_up, axis=None))
        #     surfd = surface_dist(y_true, y_pred_up, sampling=spacing)
        #     hauss_dist.append(surfd.max())
        #     mean_surf_dist.append(surfd.mean())
        #     axis = tuple(range(1, y_true.ndim))
        #     scores.append(numpy_dice(y_true, y_pred_up, axis=axis))
        #
        # ravd = np.array(ravd)
        # vol_scores = np.array(vol_scores)
        # scores = np.concatenate(scores, axis=0)
        #
        # print('Mean volumetric DSC:', vol_scores.mean())
        # print('Median volumetric DSC:', np.median(vol_scores))
        # print('Std volumetric DSC:', vol_scores.std())
        # print('Mean Hauss. Dist:', np.mean(hauss_dist))
        # print('Mean MSD:', np.mean(mean_surf_dist))
        # print('Mean Rel. Abs. Vol. Diff:', ravd.mean())

        return

    def get_data(self):
        return self._data
