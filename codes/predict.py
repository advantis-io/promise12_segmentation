import os

import cv2
from keras.optimizers import Adam
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from skimage.exposure import equalize_adapthist

from augmenters import smooth_images
from metrics import dice_coef, dice_coef_loss
from models import UNet


def get_model(img_rows, img_cols):
    dirname = '../data'
    model = UNet((img_rows, img_cols, 1), start_ch=8, depth=2, batchnorm=True,
                 dropout=0.5, maxpool=True, residual=True)
    filename = [os.path.join(dirname, f) for f in os.listdir(dirname)
                if f.startswith('weights') and f.endswith('.h5')]
    filename = filename[0]
    model.load_weights(filename)
    model.compile(optimizer=Adam(), loss=dice_coef_loss, metrics=[dice_coef])
    return model


def read_and_return_niis(path):
    mhdfiles = [os.path.join(path, f) for f in os.listdir(path)
                if f.endswith('.mhd')]

    niis = {}
    for mhdfilename in mhdfiles:
        itk_img = sitk.ReadImage(mhdfilename)
        itk_data = sitk.GetArrayFromImage(itk_img)
        # Get affine transform in LPS
        c = np.array([itk_img.TransformContinuousIndexToPhysicalPoint(p)
                      for p in ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, 0))])
        affine = np.transpose(np.concatenate([np.concatenate(
            [c[0:3] - c[3:], c[3:]], 0), [[0.], [0.], [0.], [1.]]], 1))
        # Convert to RAS to match nibabel
        affine = np.matmul(np.diag([-1., -1., 1., 1.]), affine)
        nii = nib.Nifti1Image(itk_data.T, affine)

        filename = mhdfilename[:-3] + 'nii.gz'
        niis[filename] = nii

    return niis


def return_niis(path):
    niis = {}
    for fname in os.listdir(path):
        if not (fname.endswith('.nii') or fname.endswith('.nii.gz')):
            continue
        fpath = os.path.join(path, fname)
        niis[fpath] = nib.load(fpath)
    return niis


def img_resize(imgs, img_rows, img_cols, equalize=True):
    if imgs.dtype == np.float:
        imgs = (imgs - imgs.min()) / (imgs.max() - imgs.min())

    new_imgs = np.zeros([len(imgs), img_rows, img_cols])
    for mm, img in enumerate(imgs):
        if equalize:
            img = equalize_adapthist(img, clip_limit=0.05)

        new_imgs[mm] = cv2.resize(img, (img_rows, img_cols),
                                  interpolation=cv2.INTER_NEAREST)

    return new_imgs


def resize_pred_to_val(y_pred, shape):
    row = shape[2]
    col = shape[1]

    resized_pred = np.zeros(shape)
    for mm in range(len(y_pred)):
        resized_pred[mm, :, :] = cv2.resize(y_pred[mm, :, :, 0], (row, col),
                                            interpolation=cv2.INTER_NEAREST)

    return resized_pred


def predict(niis, img_rows, img_cols, mu, sigma):
    model = get_model(img_rows, img_cols)

    for fname in niis:
        print("Predicting: {}".format(fname))
        nii = niis[fname]
        data = nii.get_data()
        # Flip from X, Y, Z to Z, Y, Z
        data_T = data.T

        slices = img_resize(data_T, img_rows, img_cols)
        slices = slices.reshape(-1, img_rows, img_cols, 1)

        slices = smooth_images(slices)
        slices = (slices - mu) / sigma

        assert (slices != 0).any()
        print(slices.sum())

        prediction = model.predict(slices, verbose=1, batch_size=1)
        if prediction.sum() == 0:
            print('{}: all 0'.format(fname))
            continue

        prediction = resize_pred_to_val(prediction, data_T.shape)

        # flip back
        prediction = prediction.T

        prediction_nii = nib.Nifti1Image(prediction, nii.affine)

        if not os.path.exists(fname):
            nii.to_filename(fname)
        if fname.endswith('.nii.gz'):
            name = fname[:-7]
        else:
            name = fname[:-4]
        prediction_nii.to_filename(name + '_prediction.nii.gz')


if __name__ == '__main__':
    input_dir = '../train_data'
    input_dir = '../papan_t2s'
    read_mhd = False

    img_rows, img_cols = 128, 128
    mu, sigma = 0.4050441030286516, 0.23546992571535455

    if read_mhd:
        niis = read_and_return_niis(input_dir)
    else:
        niis = return_niis(input_dir)

    predict(niis, img_rows, img_cols, mu, sigma)
