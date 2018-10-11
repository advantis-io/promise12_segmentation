from keras import callbacks

from metrics import rel_abs_vol_diff, surface_dist
from test import resize_pred_to_val, numpy_dice, read_cases
import numpy as np

from train import smooth_images


class MetricsCallback(callbacks.Callback):
    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set

    def on_train_begin(self, logs={}):
        imgs = []
        masks = []
        for data_obj in self.train_set:
            imgs.append(data_obj.image)
            masks.append(data_obj.mask)

        img_rows = 256
        img_cols = 256

        imgs = np.concatenate(imgs, axis=0).reshape(-1, img_rows, img_cols, 1)
        masks = np.concatenate(masks, axis=0).reshape(-1, img_rows, img_cols, 1)
        self.y_train = masks.astype(int)

        # Smooth images using CurvatureFlow
        imgs = smooth_images(imgs)

        mu = np.mean(imgs)
        sigma = np.std(imgs)
        self.X_train = (imgs - mu) / sigma


    def on_epoch_end(self, batch, logs={}):
        y_pred = self.model.predict(self.X_train, verbose=1, batch_size=128)
        print('Results on train set:')
        print('Accuracy:', numpy_dice(self.y_train, y_pred))

        y_pred = self.model.predict(self.X_test, verbose=1, batch_size=128)
        print('Results on validation set')
        print('Accuracy:', numpy_dice(self.y_test, y_pred))

        vol_scores = []
        ravd = []
        scores = []
        hauss_dist = []
        mean_surf_dist = []

        start_ind = 0
        end_ind = 0
        for data_obj in self.train_set:
            y_true = data_obj.mask
            spacing = data_obj.mask_spacing

            start_ind = end_ind
            end_ind += len(y_true)

            y_pred_up = resize_pred_to_val(y_pred[start_ind:end_ind], y_true.shape)

            ravd.append(rel_abs_vol_diff(y_true, y_pred_up))
            vol_scores.append(numpy_dice(y_true, y_pred_up, axis=None))
            surfd = surface_dist(y_true, y_pred_up, sampling=spacing)
            hauss_dist.append(surfd.max())
            mean_surf_dist.append(surfd.mean())
            axis = tuple(range(1, y_true.ndim))
            scores.append(numpy_dice(y_true, y_pred_up, axis=axis))

        ravd = np.array(ravd)
        vol_scores = np.array(vol_scores)
        scores = np.concatenate(scores, axis=0)

        print('Mean volumetric DSC:', vol_scores.mean())
        print('Median volumetric DSC:', np.median(vol_scores))
        print('Std volumetric DSC:', vol_scores.std())
        print('Mean Hauss. Dist:', np.mean(hauss_dist))
        print('Mean MSD:', np.mean(mean_surf_dist))
        print('Mean Rel. Abs. Vol. Diff:', ravd.mean())

        return

    def get_data(self):
        return self._data
