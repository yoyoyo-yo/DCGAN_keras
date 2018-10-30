# DCGAN_keras

This is ***unofficial*** DCGAN implemented with Keras.

MNIST examples ... Success!!
Cifar10, Original ... Unsuccess :(

If you succeed, please tell me!!

## Requirements

You can get python packages.

I recommed using Python3.

```bash
$ pip install -r requirements.txt
```

```bash
python 3.6 (maybe Python-2.* possible)
Tensorflow 1.9
Keras 2.2
Numpy
Matplotlib
OpenCV-python
```

## MNIST example

You can use MNIST example.

When training,

```bash
$ python main_mnist.py --train
```

The trained models (Generator and Discreminator) are stored in *models* directory.(This directory will be made automatically)

You can change above directory and other path in *config.py*

Generated images in training process are stored in *train_images* directory. (This path is defined in *config.py*)

When testing,

```bash
$ python main_mnist.py --test
```

Generated images are stored in *test_images* directory. (This pass is defined in *config.py*)


[Result] Iteration = 5000

![alt tag](mnist_result.gif)

## Cifar10 example

This is work-in-progress (unsuccessful).

When training,

```bash
$ python main_cifar10.py --train
```

When testing,

```bash
$ python main_cifar10.py --test
```

## Original data training

This is work-in-progress.

You collected images you want to generate in a directory.
You can change directory path **Train_dirs** in *config.py*.
You can set one or more passes.
For example, 
```python
Train_dirs = [
    '/mnt/c/Users/demo/Research_nagayosi/Dataset/Moca',
    '/home/usrs/demo/Dataset/Moca',
]
```

When training,

```bash
$ python main.py --train
```

When testin,

```bash
$ python main.py --test
```
