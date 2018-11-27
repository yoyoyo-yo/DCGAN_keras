#!/usr/bin/env python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

import argparse
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

import config_mnist as cf
from model_mnist import *

np.random.seed(cf.Random_seed)

from keras.datasets import mnist
Height, Width = 28, 28
Channel = 1

class Main_train():
    def __init__(self):
        pass

    def train(self):
        ## Load network model
        g = G_model(Height=Height, Width=Width, channel=Channel)
        d = D_model(Height=Height, Width=Width, channel=Channel)
        c = Combined_model(g=g, d=d)

        g.summary()

        g_opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
        d_opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)

        g.compile(loss='binary_crossentropy', optimizer='SGD')
        d.trainable = False
        for layer in d.layers:
            layer.trainable = False
        c.compile(loss='binary_crossentropy', optimizer=g_opt)
        d.trainable = True
        for layer in d.layers:
            layer.trainable = True
        d.compile(loss='binary_crossentropy', optimizer=d_opt)

        ## Prepare Training data
        #dl_train = DataLoader(phase='Train', shuffle=True)
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = (X_train.astype(np.float32) - 127.5)/127.5
        X_train = X_train[:, :, :, None]
        train_num = X_train.shape[0]
        train_num_per_step = train_num // cf.Minibatch
        data_inds = np.arange(train_num)
        max_ite = cf.Minibatch * train_num_per_step
        
        ## Start Train
        print('-- Training Start!!')
        if cf.Save_train_combine is None:
            print("generated image will not be stored")
        elif cf.Save_train_combine is True:
            print("generated image write combined >>", cf.Save_train_img_dir)
        elif cf.Save_train_combine is False:
            print("generated image write separately >>", cf.Save_train_img_dir)

        fname = os.path.join(cf.Save_dir, 'loss.txt')
        f = open(fname, 'w')
        f.write("Iteration,G_loss,D_loss{}".format(os.linesep))

        for ite in range(cf.Iteration):
            ite += 1
            # Discremenator training
            #y = dl_train.get_minibatch(shuffle=True)
            train_ind = ite % (train_num_per_step - 1)
            if ite % (train_num_per_step + 1) == max_ite:
                np.random.shuffle(data_inds)

            _inds = data_inds[train_ind * cf.Minibatch: (train_ind+1) * cf.Minibatch]
            x_fake = X_train[_inds]
            
            z = np.random.uniform(-1, 1, size=(cf.Minibatch, 100))
            #input_noise = np.random.normal(0, 0.3, size=(cf.Minibatch, 100))
            x_real = g.predict([z], verbose=0)
            x = np.concatenate((x_fake, x_real))
            t = [1] * cf.Minibatch + [0] * cf.Minibatch
            d_loss = d.train_on_batch(x, t)
            
            # Generator training
            z = np.random.uniform(-1, 1, size=(cf.Minibatch, 100))
            #input_noise = np.random.normal(0, 0.3, size=(cf.Minibatch, 100))
            g_loss = c.train_on_batch([z], [1] * cf.Minibatch)

            con = '|'
            if ite % cf.Save_train_step != 0:
                for i in range(ite % cf.Save_train_step):
                    con += '>'
                for i in range(cf.Save_train_step - ite % cf.Save_train_step):
                    con += ' '
            else:
                for i in range(cf.Save_train_step):
                    con += '>'
            con += '| '
            con += "Ite:{}, g: {:.6f}, d: {:.6f} ".format(ite, g_loss, d_loss)
            sys.stdout.write("\r"+con)

            if ite % cf.Save_train_step == 0 or ite == 1:
                print()
                f.write("{},{},{}{}".format(ite, g_loss, d_loss, os.linesep))
                # save weights
                d.save_weights(cf.Save_d_path)
                g.save_weights(cf.Save_g_path)

                gerated = g.predict([z], verbose=0)
                # save some samples
                if cf.Save_train_combine is True:
                    save_images(gerated, index=ite, dir_path=cf.Save_train_img_dir)
                elif cf.Save_train_combine is False:
                    save_images_separate(gerated, index=ite, dir_path=cf.Save_train_img_dir)
        f.close()
        ## Save trained model
        d.save_weights(cf.Save_d_path)
        g.save_weights(cf.Save_g_path)
        print('Model saved -> ', cf.Save_d_path, cf.Save_g_path)

class Main_test():
    def __init__(self):
        pass

    def test(self):
        ## Load network model
        g = G_model(Height=Height, Width=Width, channel=Channel)
        g.load_weights(cf.Save_g_path, by_name=True)

        print('-- Test start!!')
        if cf.Save_test_combine is None:
            print("generated image will not be stored")
        elif cf.Save_test_combine is True:
            print("generated image write combined >>", cf.Save_test_img_dir)
        elif cf.Save_test_combine is False:
            print("generated image write separately >>", cf.Save_test_img_dir)
        pbar = tqdm(total=cf.Test_num)

        for i in range(cf.Test_num):
            input_noise = np.random.uniform(-1, 1, size=(cf.Test_Minibatch, 100))
            g_output = g.predict(input_noise, verbose=0)

            if cf.Save_test_combine is True:
                save_images(g_output, index=i, dir_path=cf.Save_test_img_dir)
            elif cf.Save_test_combine is False:
                save_images_separate(g_output, index=i, dir_path=cf.Save_test_img_dir)
            pbar.update(1)


def save_images(imgs, index, dir_path):
    # Argment
    #  img_batch = np.array((batch, height, width, channel)) with value range [-1, 1]
    B, H, W, C = imgs.shape
    batch= imgs * 127.5 + 127.5
    batch = batch.astype(np.uint8)
    w_num = np.ceil(np.sqrt(B)).astype(np.int)
    h_num = int(np.ceil(B / w_num))
    out = np.zeros((h_num*H, w_num*W), dtype=np.uint8)
    for i in range(B):
        x = i % w_num
        y = i // w_num
        out[y*H:(y+1)*H, x*W:(x+1)*W] = batch[i, ..., 0]
    fname = str(index).zfill(len(str(cf.Iteration))) + '.jpg'
    save_path = os.path.join(dir_path, fname)

    if cf.Save_iteration_disp:
        plt.imshow(out, cmap='gray')
        plt.title("iteration: {}".format(index))
        plt.axis("off")
        plt.savefig(save_path)
    else:
        cv2.imwrite(save_path, out)


def save_images_separate(imgs, index, dir_path):
    # Argment
    #  img_batch = np.array((batch, height, width, channel)) with value range [-1, 1]
    B, H, W, C = imgs.shape
    batch= imgs * 127.5 + 127.5
    batch = batch.astype(np.uint8)
    for i in range(B):
        save_path = os.path.join(dir_path, '{}_{}.jpg'.format(index, i))
        cv2.imwrite(save_path, batch[i])


def arg_parse():
    parser = argparse.ArgumentParser(description='CNN implemented with Keras')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()

    if args.train:
        main = Main_train()
        main.train()
    if args.test:
        main = Main_test()
        main.test()

    if not (args.train or args.test):
        print("please select train or test flag")
        print("train: python main.py --train")
        print("test:  python main.py --test")
        print("both:  python main.py --train --test")
