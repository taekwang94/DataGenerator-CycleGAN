from __future__ import print_function, division
import scipy
import scipy.misc
from PIL import Image
from keras.datasets import mnist
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
from keras_contrib.layers.normalization import instancenormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import keras_contrib
import datetime
import matplotlib.pyplot as plt
import sys
import time
import json
import math
import csv
from data_loader import DataLoader
import numpy as np
import os
from data_loader import DataLoader
from keras.utils.training_utils import multi_gpu_model
from keras.models import load_model
from keras.models import model_from_json
from keras.models import load_model
import cv2
import argparse

def parser_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--load_model_path', default='/disk2/taekwang/GANresult/saved_model')
    parser.add_argument('--dataset_path', default='/disk2/taekwang/GANdataset')
    parser.add_argument('--dataset_name', default='fogdata_1')
    parser.add_argument('--data_save_path', default='/disk2/taekwang/GANresult')
    parser.add_argument('--modelEpoch_num', default=10)
    parser.add_argument('--isConcat', default=False)


    return parser.parse_args()

img_rows = 256
img_cols = 256

img_rows_re = 224
img_cols_re = 224

class DataGenerator():
    def __init__(self, rows, cols, modelEpochNumber):
        self.args = parser_args()
        self.rows = rows
        self.cols = cols
        self.modelEpochNumber = modelEpochNumber
        self.d_A , self.d_B, self.g_AB, self.g_BA = self.load_Trained_model(self.modelEpochNumber)


    def load_Trained_model(self, modelEpochNumber):
        d_A = load_model(os.path.join(self.args.load_model_path, 'd_A_{}.h5').format(modelEpochNumber),
                         custom_objects={'InstanceNormalization': keras_contrib.layers.InstanceNormalization})
        d_B = load_model(os.path.join(self.args.load_model_path, 'd_B_{}.h5').format(modelEpochNumber),
                         custom_objects={'InstanceNormalization': keras_contrib.layers.InstanceNormalization})
        g_AB = load_model(os.path.join(self.args.load_model_path, 'g_AB_{}.h5').format(modelEpochNumber),
                          custom_objects={'InstanceNormalization': keras_contrib.layers.InstanceNormalization})
        g_BA = load_model(os.path.join(self.args.load_model_path, 'g_BA_{}.h5').format(modelEpochNumber),
                          custom_objects={'InstanceNormalization': keras_contrib.layers.InstanceNormalization})
        return d_A, d_B, g_AB, g_BA

    def generate_image(self, count, batch_i, images_A, images_B, isConcat):
        os.makedirs(self.args.data_save_path+'/%s' % self.args.dataset_name, exist_ok= True)

        fake_B = self.g_AB.predict(images_A)
        fake_A = self.g_BA.predict(images_B)

        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)
        r, c = 2, 2

        if isConcat:
            gen_imgs = np.concatenate([images_A, fake_B, images_B, fake_A])
            gen_imgs = 0.5 * gen_imgs + 0.5
            titles = ['Original', 'Translated', 'Reconstructed']
            fig, axs = plt.subplot(r,c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i, j].imshow(gen_imgs[cnt])
                    axs[i, j].set_title(titles[j])
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig(self.args.data_save_path + "/concat_images/%s/%d_%d.png" % (self.args.dataset_name, count, batch_i))
            plt.close()
        else:
            fake_A_resized = cv2.resize(fake_A[0, :, :, :], (img_rows_re, img_cols_re), interpolation=cv2.INTER_CUBIC)
            fake_B_resized = cv2.resize(fake_B[0, :, :, :], (img_rows_re, img_cols_re), interpolation=cv2.INTER_CUBIC)

            normal_save_path = self.args.data_save_path +  "/%s/normal" % self.args.dataset_name
            fog_save_path = self.args.data_save_path + "/%s/fog" % self.args.dataset_name

            normal_save_image = self.args.data_save_path + "/%s/normal/fake_normal_%d_%d.png" % (self.args.dataset_name, count, batch_i)
            fog_save_image = self.args.data_save_path + "/%s/fog/fake_fog_%d_%d.png" % (self.args.dataset_name, count, batch_i)

            if not os.path.exists(normal_save_path):
                os.makedirs(normal_save_path)
            if not os.path.exists(fog_save_path):
                os.makedirs(fog_save_path)

            scipy.misc.imsave(normal_save_image, fake_A_resized)
            scipy.misc.imsave(fog_save_image, fake_B_resized)




def main():
    args = parser_args()
    datagen = DataGenerator(img_rows, img_cols, args.modelEpoch_num)

    data_loader = DataLoader(dataset_path=args.dataset_path ,dataset_name=args.dataset_name, img_res=(img_rows, img_cols))
    print(data_loader)

    cnt_ = 0

    for batch_i, (imgs_A, imgs_B) in enumerate(data_loader.load_batch(1)):
        print("Count :", cnt_)
        datagen.generate_image(count= batch_i, batch_i = 1, images_A=imgs_A, images_B=imgs_B, isConcat=args.isConcat)





if __name__ == "__main__":
    main()