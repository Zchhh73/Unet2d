from __future__ import division
from GlandCeil_Unet.dilated_unet.model import dilatedUnet2dModule
import numpy as np
import pandas as pd
import cv2


def train():
    '''
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    csvmaskdata = pd.read_csv('H:\\py_workplace\\Unet2d\\GlandCeil_Unet\\train_mask.csv')
    csvimagedata = pd.read_csv('H:\\py_workplace\\Unet2d\\GlandCeil_Unet\\train_img.csv')
    maskdata = csvmaskdata.iloc[:, :].values
    imagedata = csvimagedata.iloc[:, :].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(csvimagedata))
    np.random.shuffle(perm)
    imagedata = imagedata[perm]
    maskdata = maskdata[perm]

    unet2d = dilatedUnet2dModule(512, 512, channels=3, costname="dice coefficient")
    unet2d.train(imagedata, maskdata, "H:\\py_workplace\\Unet2d\\GlandCeil_Unet\\model",
                 "H:\\py_workplace\\Unet2d\\GlandCeil_Unet\\log", 0.0005, 0.8, 100000, 2)


def main(argv):
    if argv == 1:
        train()
    # if argv == 2:
    #     predict()


if __name__ == '__main__':
    main(1)
