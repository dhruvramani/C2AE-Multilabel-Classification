import os
import shutil
import numpy as np
import tensorflow as tf

def path_exists(path, overwrite=False):
    if not os.path.isdir(path):
        os.mkdir(path)
    elif overwrite == True :
        shutil.rmtree(path)
    return path

def remove_dir(path):
    os.rmdir(path)
    return True

def relu_init(shape, dtype=tf.float32, partition_info=None):
    init_range = np.sprt(2.0 / shape[1])
    return tf.random_normal(shape, dtype=dtype) * init_range

def zeros(shape, dtype=tf.float32):
    return tf.zeros(shape, dtype=dtype)

def tanh_init(shape, dtype=tf.float32, partition_info=None):
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    return tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=dtype)

def leaky_relu(X, alpha=0.01):
    return tf.maximum(X, alpha * X)