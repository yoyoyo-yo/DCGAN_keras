# DCGAN_keras

This is ***unofficial*** DCGAN implemented with Keras.

**MNIST, Cifar10 ... Success!!**

**Original ... Unsuccess :(** (work in progress)

***If you succeed, please tell me!!***

## Requirements

You can get python packages.

I recommend using Python3, and ***strongly recommend GPU when training (cifar10, original your own dataset with large iteration, minibatch and size ( >= 32x32).***

When using CPU

```bash
$ pip install -r requirements.txt
```

When using GPU

```bash
$ pip install -r requirements_gpu.txt
```

## MNIST example

You can use MNIST example.

When training,

```bash
$ python main_mnist.py --train
```

The trained models (Generator and Discreminator) are stored in **models_mnist** directory.(This directory will be made automatically)

You can change above directory and other path in **config_mnist.py**

Generated images in training process are stored in **train_images_mnist** directory (This path is defined in **config_mnist.py**), and the ones in test are stored in **test_images_mnist**

The models are defined in **model_mnist.py**

When testing,

```bash
$ python main_mnist.py --test
```

Generated images are stored in **test_images** directory. (This pass is defined in **config.py**)


|Training process|Iteration 3000, 28x28|
|---|---|
|![](assets/mnist_result.gif)|![](assets/mnist_3000.jpg)|

## Cifar10 example

The models are defined by **model_cifar10.py**

Generated images in training process are stored in **train_image_cifar10** (defined in **config_cifar10.py**)

Generated images in test are stored in **test_image_cifar10** (defined in **config_cifar10.py**)

|Training process|Iteration 30000, 32x32|
|---|---|
|![](assets/cifar10_result.gif)|![](assets/cifar10_30000.jpg)|

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
You can change directory path **Train\_dirs** in **config.py**.
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
