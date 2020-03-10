#import tensorflow as tf
import numpy as np

import vgg

from tensorflow import keras
#from data import IIC_ImageDataGenerator
from data_generator import DataGenerator

from utils import AccuracyCallback, lr_schedule, unsupervised_labels, center_crop


class IIC_clustering:
    def __init__(self,
                 input_shape,
                 path,
                 heads,
                 batch_size,
                 epochs,
                 classes):
        """Contains the encoder model, the loss function,
            loading of datasets, train and evaluation routines
            to implement IIC unsupervised clustering

        Arguments:
            args : Command line arguments to indicate choice
                of batch size, number of heads, folder to save
                weights file, weights file name, etc
            backbone (Model): IIC Encoder backbone (eg VGG):
        """
        #self.args = args
        #self.backbone = backbone
        self._model = None
        self.x_test = None
        self.n_labels = 10

        # self.load_eval_dataset()
        self.accuracy = 0
        self.input_shape = input_shape
        self.path = path
        self.heads = heads
        self.batch_size = batch_size
        self.epochs = epochs
        self.classes = classes

        self.train_gen = DataGenerator(
            2, shuffle=True, siamese=True, batch_size=self.batch_size)
        self.build_model()
        self.load_eval_dataset()

    def build_model(self):
        """Build the n_heads of the IIC model
        """
        # construct CNN model
        self.CNN = vgg.VGG(vgg.cfg['F'], self.input_shape).model

        inputs = keras.layers.Input(shape=self.input_shape, name='x')
        x = self.CNN(inputs)
        x = keras.layers.Flatten()(x)
        # number of output heads
        outputs = []
        for i in range(self.heads):
            name = "z_head%d" % i
            outputs.append(keras.layers.Dense(self.n_labels,
                                              activation='softmax',
                                              name=name)(x))
        self._model = keras.models.Model(inputs, outputs, name='encoder')
        optimizer = keras.optimizers.Adam(lr=1e-3)
        self._model.compile(optimizer=optimizer, loss=self.mi_loss)
        self._model.summary()

    def fit(self):
        """Train function uses the data generator,
            accuracy computation, and learning rate
            scheduler callbacks
        """
        accuracy = AccuracyCallback(self)
        lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule,
                                                             verbose=1)
        callbacks = [accuracy, lr_scheduler]
        self._model.fit_generator(generator=self.train_gen,
                                  use_multiprocessing=False,
                                  epochs=self.epochs,
                                  callbacks=callbacks,
                                  workers=4,
                                  shuffle=True)

    def mi_loss(self, y_true, y_pred):
        """Mutual information loss computed from the joint
           distribution matrix and the marginals

        @param y_true (tensor): Not used because of unsupervised
                learning. However, is required to be used with fit method from
                keras model class

        @param y_pred (tensor): stack of softmax predictions for
                the latent vectors (Z and Zbar)
        """
        size = self.batch_size
        n_labels = y_pred.shape[-1]
        # lower half is Z
        Z = y_pred[0: size, :]
        Z = keras.backend.expand_dims(Z, axis=2)
        # upper half is Zbar
        Zbar = y_pred[size: y_pred.shape[0], :]
        Zbar = keras.backend.expand_dims(Zbar, axis=1)
        # compute joint distribution
        P = keras.backend.batch_dot(Z, Zbar)
        P = keras.backend.sum(P, axis=0)
        # enforce symmetric joint distribution
        P = (P + keras.backend.transpose(P)) / 2.0
        # normalization of total probability to 1.0
        P = P / keras.backend.sum(P)
        # marginal distributions
        Pi = keras.backend.expand_dims(keras.backend.sum(P, axis=1), axis=1)
        Pj = keras.backend.expand_dims(keras.backend.sum(P, axis=0), axis=0)
        Pi = keras.backend.repeat_elements(Pi, rep=n_labels, axis=1)
        Pj = keras.backend.repeat_elements(Pj, rep=n_labels, axis=0)
        P = keras.backend.clip(P, keras.backend.epsilon(), np.finfo(float).max)
        Pi = keras.backend.clip(
            Pi, keras.backend.epsilon(), np.finfo(float).max)
        Pj = keras.backend.clip(
            Pj, keras.backend.epsilon(), np.finfo(float).max)
        # negative MI loss
        neg_mi = keras.backend.sum(
            (P * (keras.backend.log(Pi) + keras.backend.log(Pj) - keras.backend.log(P))))
        # each head contribute 1/n_heads to the total loss
        return neg_mi/self.heads

    def eval(self):
        """Evaluate the accuracy of the current model weights
        """
        y_pred = self._model.predict(self.x_test)
        print("")
        # accuracy per head
        for head in range(self.heads):
            if self.heads == 1:
                y_head = y_pred
            else:
                y_head = y_pred[head]
            y_head = np.argmax(y_head, axis=1)

            accuracy = unsupervised_labels(list(self.y_test),
                                           list(y_head),
                                           self.n_labels,
                                           self.n_labels)
            info = "Head %d accuracy: %0.2f%%"
            if self.accuracy > 0:
                info += ", Old best accuracy: %0.2f%%"
                data = (head, accuracy, self.accuracy)
            else:
                data = (head, accuracy)
            print(info % data)
            # if accuracy improves during training,
            # save the model weights on a file
            # if accuracy > self.accuracy \
            #         and self.args.save_weights is not None:
            #     self.accuracy = accuracy
            #     folder = self.args.save_dir
            #     os.makedirs(folder, exist_ok=True)
            #     path = os.path.join(folder, self.args.save_weights)
            #     print("Saving weights... ", path)
            #     self._model.save_weights(path)

    def load_eval_dataset(self):
        """Pre-load test data for evaluation
        """
        
        (_, _), (x_test, self.y_test) = keras.datasets.mnist.load_data()
        image_size = x_test.shape[1]
        x_test = np.reshape(x_test,[-1, image_size, image_size, 1])
        x_test = x_test.astype('float32') / 255
        x_eval = np.zeros([x_test.shape[0], *self.train_gen.input_shape])
        for i in range(x_eval.shape[0]):
            x_eval[i] = center_crop(x_test[i])

        self.x_test = x_eval

    @property
    def model(self):
        return self._model
