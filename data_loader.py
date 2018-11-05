import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

import config as cf

Dataset_Debug_Display = False

class DataLoader():
    def __init__(self, phase='Train', shuffle=False):
        self.datas = []
        self.last_mb = 0
        self.phase = phase
        self.prepare_datas(shuffle=shuffle)

    def prepare_datas(self, shuffle=True):
        if self.phase == 'Train':
            dir_paths = cf.Train_dirs
        elif self.phase == 'Test':
            dir_paths = cf.Test_dirs
        print('------------\nData Load (phase: {})'.format(self.phase))

        for dir_path in dir_paths:
            files = []
            for ext in cf.File_extensions:
                files += glob.glob(dir_path + '/*' + ext)
                
            load_count = 0
            for img_path in files:
                if cv2.imread(img_path) is None:
                    continue
                data = {'img_path': img_path,
                        'gt_path': 0,
                        'h_flip': False,
                        'v_flip': False,
                        'rotate': False
                }
                self.datas.append(data)
                load_count += 1

            print(' - {} - {} datas -> loaded {}'.format(dir_path, len(files), load_count))

        self.display_gt_statistic()
        if self.phase == 'Train':
            self.data_augmentation()
            self.display_gt_statistic()

        self.set_index(shuffle=shuffle)

    def display_gt_statistic(self):
        print(' -*- Training label  -*-')
        print('   Total data: {}'.format(len(self.datas)))

    def set_index(self, shuffle=True):
        self.data_n = len(self.datas)
        self.indices = np.arange(self.data_n)
        if shuffle:
            np.random.seed(cf.Random_seed)
            np.random.shuffle(self.indices)

    def get_minibatch_index(self, shuffle=False):
        if self.phase == 'Train':
            mb = cf.Minibatch
        elif self.phase == 'Test':
            mb = 1
        _last = self.last_mb + mb
        if _last >= self.data_n:
            mb_inds = self.indices[self.last_mb:]
            self.last_mb = _last - self.data_n
            if shuffle:
                np.random.seed(cf.Random_seed)
                np.random.shuffle(self.indices)
            _mb_inds = self.indices[:self.last_mb]
            mb_inds = np.hstack((mb_inds, _mb_inds))
        else:
            mb_inds = self.indices[self.last_mb : self.last_mb+mb]
            self.last_mb += mb

        self.mb_inds = mb_inds

    def get_minibatch(self, shuffle=True):
        if self.phase == 'Train':
            mb = cf.Minibatch
        elif self.phase == 'Test':
            mb = 1
        self.get_minibatch_index(shuffle=shuffle)
        imgs = np.zeros((mb, cf.Height, cf.Width, cf.Channel), dtype=np.float32)

        for i, ind in enumerate(self.mb_inds):
            data = self.datas[ind]
            img = self.load_image(data['img_path'])
            img = self.image_dataAugment(img, data)

            imgs[i] = img
            #print(data['img_path'], gt)
            if Dataset_Debug_Display:
                print(data['img_path'])
                print()
                plt.imshow(imgs[i].transpose(1,2,0))
                plt.subplots()
                plt.imshow(gts[i,0])
                plt.show()

        if cf.Input_type == 'channels_first':
            imgs = imgs.transpose(0, 3, 1, 2)

        return imgs

    ## Below functions are for data augmentation
    def load_image(self, img_name):
        if cf.Channel == 3:
            img = cv2.imread(img_name)
            img = img[..., (2,1,0)]
        elif cf.Channel == 1:
            img = cv2.imread(img_name, 0)
        else:
            raise Exception("Channel Error in config.py")
        img = cv2.resize(img, (cf.Width, cf.Height))
        img = img.astype(np.float32) / 127.5 - 1.
        #img = img / 255.
        if cf.Channel == 1:
            img = img[..., None]
        return img

    def image_dataAugment(self, image, data):
        if data['h_flip']:
            image = image[:, ::-1]
        if data['v_flip']:
            image = image[::-1, :]
        if data['rotate']:
            max_side = max(h, w)
            if len(image.shape) == 3:
                frame = np.zeros((max_side, max_side, 3), dtype=np.float32)
            elif len(image.shape) == 2:
                frame = np.zeros((max_side, max_side), dtype=np.float32)
            tx = int((max_side-w)/2)
            ty = int((max_side-h)/2)
            frame[ty:ty+h, tx:tx+w] = image
            M = cv2.getRotationMatrix2D((max_side/2, max_side/2), 90, 1)
            rot = cv2.warpAffine(frame, M, (max_side, max_side))
            image = rot[tx:tx+w, ty:ty+h]
        return image

    def data_augmentation(self):
        print('   |   -*- Data Augmentation -*-')
        if cf.Horizontal_flip:
            self.add_horizontal_flip()
            print('   |    - Added horizontal flip')
        if cf.Vertical_flip:
            self.add_vertical_flip()
            print('   |    - Added vertival flip')
        if cf.Rotate_ccw90:
            self.add_rotate_ccw90()
            print('   |    - Added Rotate ccw90')
        print('   \/')


    def add_horizontal_flip(self):
        new_data = []
        for data in self.datas:
            _data = data.copy()
            _data['h_flip'] = True
            new_data.append(_data)
        self.datas.extend(new_data)

    def add_vertical_flip(self):
        new_data = []
        for data in self.datas:
            _data = data.copy()
            _data['v_flip'] = True
            new_data.append(_data)
        self.datas.extend(new_data)

    def add_rotate_ccw90(self):
        new_data = []
        for data in self.datas:
            _data = data.copy()
            _data['rotate'] = True
            new_data.append(_data)
        self.datas.extend(new_data)
