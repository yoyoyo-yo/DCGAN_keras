import os

## Network config
##  Input width and Height
#Height, Width = 28, 28  # MNIST
#Height, Width = 64, 64
#Channel = 3

# input data shape
# channels_last -> [mb, c, h, w] , channels_first -> [mb, h, w, c]
Input_type = 'channels_last'

## Directory paths for training
#Train_dirs = [
#    '/mnt/c/Users/demo/Research_nagayosi/Dataset/Moca',
#    '/mnt/c/Users/demo/Research_nagayosi/Dataset/text_image'
#]

## Data augmentation
#Horizontal_flip = True
#Vertical_flip = False
#Rotate_ccw90 = False

File_extensions = ['.jpg', '.png']

## Training config
Iteration = 10000
Minibatch = 64

## Test config
## The total number of generated images is Test_Minibatch * Test_num
Test_Minibatch = 100
Test_num = 10
Save_test_img_dir = 'test_images_cifar10'

# if Save_combine is True, generated images in test are stored combined with same minibatch's
# if False, generated images are stored separately
# if None, generated image is not stored
Save_train_combine = True
Save_test_combine = True

Save_train_step = 20
Save_iteration_disp = True

## Save config
Save_dir = 'models_cifar10'
Save_d_name = 'D.h5'
Save_g_name = 'G.h5'
Save_d_path = os.path.join(Save_dir, Save_d_name)
Save_g_path = os.path.join(Save_dir, Save_g_name)
Save_train_img_dir = 'train_images_cifar10'
Save_img_num = 5

## Other config
##  Randon_seed is used for seed of dataset shuffle in data_loader.py
Random_seed = 0

## Check
variety = ['channels_first', 'channels_last']
if not Input_type in variety:
    raise Exception("unvalid Input_type")

#os.system("rm {}/*".format(Save_train_img_dir))
import platform
python_version = int(platform.python_version_tuple()[0])
if python_version == 3:
    os.makedirs(Save_dir, exist_ok=True)
    os.makedirs(Save_train_img_dir, exist_ok=True)
    os.makedirs(Save_test_img_dir, exist_ok=True)
elif python_version == 2:
    if not os.path.exists(Save_dir):
        os.makedirs(Save_dir)
    if not os.path.exists(Save_train_img_dir):
        os.makedirs(Save_train_img_dir)
    if not os.path.exists(Save_test_img_dir):
        os.makedirs(Save_test_img_dir)
