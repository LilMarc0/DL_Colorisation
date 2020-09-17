import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers import MaxPool2D, UpSampling2D, Lambda, Multiply, UpSampling2D, PReLU
from keras.layers.convolutional import Conv2D
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.merge import Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from logging import getLogger
from tensorflow import Session
from keras.backend import set_session
import hashlib
import json
from tensorflow.compat.v1.logging import set_verbosity, ERROR
from tensorflow.compat.v1.math import log_softmax


def adauga_bloc_rezidual(x, index, mc, nr_f):
    in_x = x
    nume_bloc = "res" + str(index)
    x = Conv2D(filters=nr_f, kernel_size=mc.dim_kernel, padding="same",
                 use_bias=False, kernel_regularizer=l2(mc.l2_reg),
               name=nume_bloc + "_conv1-" + str(mc.dim_kernel) + "-" + str(nr_f))(x)
    x = BatchNormalization(axis=-1, name=nume_bloc + "_batchnorm1")(x)
    x = PReLU()(x)
    x = Conv2D(filters=nr_f, kernel_size=mc.dim_kernel, padding="same",
                 use_bias=False, kernel_regularizer=l2(mc.l2_reg),
               name=nume_bloc + "_conv2-" + str(mc.dim_kernel) + "-" + str(nr_f))(x)
    x = BatchNormalization(axis=-1, name="res" + str(index) + "_batchnorm2")(x)
    x = Add(name=nume_bloc + "_add")([in_x, x])
    x = PReLU()(x)
    return x


def resAE(mc):
    in_x = x = Input((mc.width, mc.heigth, 1))
    x = Conv2D(filters=mc.starting_f, kernel_size=mc.dim_primul_kernel, padding="same",
                 use_bias=False, kernel_regularizer=l2(mc.l2_reg),
               name="input_conv-" + str(mc.dim_primul_kernel) + "-" + str(mc.starting_f))(x)
    x = BatchNormalization(axis=-1,
                           name="input_batchnorm")(x)
    x = PReLU()(x)

    idx_res = 1
    starting_f = mc.starting_f
    for i in range(mc.nr_downscale):
        for j in range(mc.nr_res):
            x = adauga_bloc_rezidual(x, idx_res, mc, starting_f)
            idx_res += 1
        x = Conv2D(filters=starting_f, kernel_size=(3, 3), strides=(2, 2), padding='same')(x)
        x = PReLU()(x)

    for i in range(mc.nr_downscale):
        for j in range(mc.nr_res):
            x = adauga_bloc_rezidual(x, idx_res, mc, starting_f)
            idx_res += 1
        x = UpSampling2D((2, 2))(x)
    res_out = Conv2D(filters=2, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='tanh')(x)
    return Model(in_x, [res_out])

class ConfiguratieModel:
    width = 256
    heigth = 256
    starting_f = 128
    nr_downscale = 3
    nr_res = 4
    dim_primul_kernel = 5
    dim_kernel = 3
    nr_bloc_rezid = 5
    l2_reg = 1e-4

mc = ConfiguratieModel()