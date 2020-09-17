import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.models import load_model
import os
# import SGD and Adam optimizers
from keras.optimizers import Adam
from DataSet import *
import numpy as np
import cv2 as cv
from keras.callbacks import ReduceLROnPlateau
from ResNext import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

class AeModel:

    def __init__(self, data_set: DataSet):
        self.data_set = data_set
        self.num_epochs = 50
        self.batch_size = 2
        self.learning_rate = 10 ** -5
        self.model = None
        self.checkpoint_dir = './checkpoints/'
        self.X_val = []
        self.Y_val = []


    def define_my_model(self):
        self.model = resAE(mc)

    def compile_the_model(self):
        # compilam modelul
        # defineste optimizatorul
        optimizer = Adam(lr=10**-6)
        # apeleaza functia 'compile' cu parametrii corespunzatori.
        self.model.compile(optimizer=optimizer, loss='mse')

    def train_the_model(self):
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=2, min_lr=10**-8)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        # definim callback-ul pentru checkpoint
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_dir + '/model.{epoch:05d}.hdf5')

        train, self.X_val, y_train, self.Y_val = train_test_split(self.data_set.input_training_images,
                                                      self.data_set.ground_truth_training_images,
                                                      test_size=0.2,
                                                      random_state=1)

        self.model.fit(train,
                       y_train,
                       epochs=self.num_epochs,
                       callbacks=[checkpoint_callback, reduce_lr],
                       batch_size=self.batch_size,
                       validation_data=(self.X_val, self.Y_val))

    def evaluate_the_model(self):
        best_epoch = self.num_epochs  # puteti incerca si cu alta epoca de exemplu cu prima epoca,
        best_model = load_model(os.path.join(self.checkpoint_dir, 'model.%05d.hdf5') % best_epoch)


        for i in range(len(self.data_set.input_test_images)):
            # prezicem canalele ab pe baza input_test_images[i]
            pred_ab = best_model.predict(np.expand_dims(self.data_set.input_test_images[i], axis=0))[0]
            a, b = cv.split(pred_ab)
            # reconstruim reprezentarea Lab
            Lab_image = cv.merge((self.data_set.input_test_images[i], a*128, b*128))
            # convertim din Lab in BGR
            pred_image = cv.cvtColor(Lab_image, cv.COLOR_LAB2BGR) * 255
            # convertim imaginea de input din L in 'grayscale'
            input_image = np.uint8(self.data_set.input_test_images[i] / 100 * 255)
            # imaginea ground-truth in format bgr
            gt_image = np.uint8(self.data_set.ground_truth_bgr_test_images[i])
            # pred_image este imaginea prezisa in format BGR.
            concat_images = self.concat_images(input_image, pred_image, gt_image)
            cv.imwrite(os.path.join(self.data_set.dir_output_images, '%d.png' % i), concat_images)

    def concat_images(self, input_image, pred, ground_truth):
        """
        :param input_image: imaginea grayscale (canalul L din reprezentarea Lab).
        :param pred: imaginea prezisa.
        :param ground_truth: imaginea ground-truth.
        :return: concatenarea imaginilor.
        """
        h, w, _ = input_image.shape
        space_btw_images = int(0.2 * h)
        image = np.ones((h, w * 3 + 2 * space_btw_images, 3)) * 255
        # add input_image
        image[:, :w] = input_image
        # add predicted
        offset = w + space_btw_images
        image[:, offset: offset + w] = pred
        # add ground truth
        offset = 2 * (w + space_btw_images)
        image[:, offset: offset + w] = ground_truth
        return np.uint8(image)