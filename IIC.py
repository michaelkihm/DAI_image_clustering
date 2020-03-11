import numpy as np
import cv2 as cv
import vgg

from tensorflow import keras
from iic_image_data_generator import IIC_ImageDataGenerator
from typing import Tuple, List

from utils import AccuracyCallback, lr_schedule, unsupervised_labels

# pylint: disable=unused-argument


class IIC_clustering:
    def __init__(self,
                 path: str,
                 heads: int,
                 batch_size: int,
                 epochs: int,
                 z_dimension: int,
                 class_list: List = (''),
                 crop_image: int = 4,
                 input_shape: Tuple = (28, 28, 3),
                 CNN_base: str = 'ResNet',
                 mnist: bool = False,):
        """
        @brief Contains the encoder model, the loss function,
            loading of datasets, train and evaluation routines
            to implement IIC unsupervised clustering
        @param path Path to image dataset
        @param heads number of heads
        @param batch size batch size
        @paramepochs number of epochs to train
        @param z_dimension output dimension of clustering layer
        @param class_list list or tuple of classes. Only required for labeled data
        @param crop_image Defines the pixel of which the image has to be cropped
        @param input_shape original input shape of the data. The image will size will be reduced if value for crop_image > 0.
        @param CNN_base base CNN to create feature space. Values: 'Vgg', 'ResNet'
        @param mnist Do we train on mnist dataset. If yes, keras data can be used to evaluate accuracy
        """
        self._model = None
        self.x_test = None
        self._input_shape = None  # see compute_image_size
        self._uncroped_image_size = None  # see compute_image_size
        self._image_size = None  # see compute_image_size
        self._accuracy = 0
        self._crop_image = crop_image
        self._path = path
        self._heads = heads
        self._batch_size = batch_size
        self.epochs = epochs
        self._class_list = class_list
        self._z_dimension = z_dimension
        self._cnn_base = CNN_base
        self.mnsit = mnist

        self.compute_image_size(input_shape)
        self.train_gen = IIC_ImageDataGenerator(
            self._path, self._class_list, heads=self._heads, batch_size=self._batch_size, image_size=self._uncroped_image_size, crop_image=self._crop_image)
        self.build_model()

        if self.mnsit:
            self.load_mnist_eval_dataset()

        self._steps_per_epoch = self.train_gen.get_dataset_size(
        )//self.train_gen.get_batch_size()

    def build_model(self):
        """
        @brief Build the n_heads of the IIC model
        """
        # construct CNN model
        # vgg.VGG(vgg.cfg['F'], self._input_shape).model
        self.CNN = self.build_base_model()
        inputs = keras.layers.Input(shape=self._input_shape, name='x')
        x = self.CNN(inputs)
        x = keras.layers.Flatten()(x)

        # number of output heads
        outputs = []
        for i in range(self._heads):
            name = "z_head%d" % i
            outputs.append(keras.layers.Dense(self._z_dimension,
                                              activation='softmax',
                                              name=name)(x))
        self._model = keras.models.Model([inputs], outputs, name='IIC')
        optimizer = keras.optimizers.Adam(lr=1e-3)
        self._model.compile(optimizer=optimizer, loss=self.mi_loss)
        self._model.summary()

    def build_base_model(self):
        """
        @brief builds CNN base model for IIC model
        @remark ResNet is pretrained ResNet model trained on imageNet
        """
        if self._cnn_base.lower() == 'vgg':
            return vgg.VGG(vgg.cfg['F'], self._input_shape).model
        elif self._cnn_base.lower() == 'resnet':
            assert self._input_shape >= (200, 200, 3), error_msg['ResNetShape']
            base_model = keras.applications.ResNet50(
                weights="imagenet", include_top=False, input_shape=self._input_shape)
            # only train conv5 block
            for layer in base_model.layers:
                if "conv5_" in layer.name:
                    break
                layer.trainable = False
            return base_model
        elif self._cnn_base.lower() == 'test':
            model = keras.models.Sequential()
            model.add(keras.layers.Conv2D(32, kernel_size=(5, 5),
                                          activation='relu',
                                          input_shape=self._input_shape))
            model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(keras.layers.Dropout(0.25))
            model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
            model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
            model.add(keras.layers.Dropout(0.25))
            return model
        elif self._cnn_base.lower() == 'mini':
            assert self._input_shape >= (200, 200, 3), error_msg['MiniNet']
            base_model = keras.applications.mobilenet.MobileNet(
                input_shape=self._input_shape, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=False, weights='imagenet')
            for layer in base_model.layers:
                if "conv_dw_9" in layer.name:
                    break
                layer.trainable = False 
            return base_model
        else:
            raise NameError('Unknown CNN model')

    def fit(self):
        """
        @brief Train function uses the data generator,
            accuracy computation, and learning rate
            scheduler callbacks
        """
        lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule,
                                                             verbose=1)
        if self.mnsit:
            accuracy = AccuracyCallback(self)
            callbacks = [accuracy, lr_scheduler]
        else:
            callbacks = [lr_scheduler]

        self._model.fit(self.train_gen,
                                  steps_per_epoch=self._steps_per_epoch,
                                  use_multiprocessing=False,
                                  epochs=self.epochs,
                                  callbacks=callbacks,
                                  workers=4,)

    def mi_loss(self, y_true, y_pred):
        """
        @brief Mutual information loss computed from the joint
                distribution matrix and the marginals

        @param y_true (tensor): Not used because of unsupervised
                learning. However, is required to be used with fit method from
                keras model class

        @param y_pred (tensor): stack of softmax predictions for
                the latent vectors (Z and Zbar)
        """
        size = self._batch_size
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
        return neg_mi/self._heads

    def eval(self):
        """
        @brief Evaluate the accuracy of the current model weights
        """
        y_pred = self._model.predict(self.x_test)
        print("")
        # accuracy per head
        for head in range(self._heads):
            if self._heads == 1:
                y_head = y_pred
            else:
                y_head = y_pred[head]
            y_head = np.argmax(y_head, axis=1)

            accuracy = unsupervised_labels(list(self.y_test),
                                           list(y_head),
                                           self._z_dimension,
                                           self._z_dimension)
            info = "Head %d accuracy: %0.2f%%"
            if self._accuracy > 0:
                info += ", Old best accuracy: %0.2f%%"
                data = (head, accuracy, self._accuracy)
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

    def load_mnist_eval_dataset(self):
        """
        @brief Pre-load test data for evaluation
        @remark Only for testing with MNIST
        """

        (_, _), (x_test, self.y_test) = keras.datasets.mnist.load_data()
        # image_size = x_test.shape[1]
        x_test_rgb = np.zeros(shape=(x_test.shape[0], 24, 24, 3))
        for i in range(len(x_test)):
            x_test_rgb[i] = cv.resize(self.to_rgb(
                x_test[i, :, :, np.newaxis]), (24, 24))
        # x_test = np.reshape(x_test,[-1, image_size, image_size, 1])
        # x_test = x_test.astype('float32') / 255
        # x_eval = np.zeros([x_test.shape[0], *self.train_gen.get_input_shape()])#*self.train_gen.input_shape])
        # for i in range(x_eval.shape[0]):
        #     x_eval[i] = center_crop(x_test[i])

        # self.x_test = x_eval
        self.x_test = np.array(x_test_rgb)

    def to_rgb(self, im):
        # I think this will be slow
        w, h, _ = im.shape
        ret = np.empty((w, h, 3), dtype=np.float32)
        ret[:, :, 0] = im[:, :, 0]
        ret[:, :, 1] = im[:, :, 0]
        ret[:, :, 2] = im[:, :, 0]
        return ret/255.

    def compute_image_size(self, input_shape: Tuple):
        """
        @brief computes the final shape of the input data. 
                Image size will change if cropping is enabled
        """
        chanels = 3
        if self._crop_image != 0:
            assert input_shape[0] > self._crop_image and input_shape[1] > self._crop_image, error_msg['large_crop']
            self._input_shape = (*tuple((input_shape[i]-self._crop_image)
                                        for i in range(2)), chanels)
        else:
            self._input_shape = input_shape
        self._image_size = self._input_shape[0:-1]
        self._uncroped_image_size = tuple(
            (i + self._crop_image) for i in self._image_size)

    @property
    def model(self):
        return self._model


error_msg = {
    'ResNetShape': "Image shape has to be >200,200 for ResNet. Do not forget that cropping does reduce image size",
    'large_crop': "Crop size is larger than inuput shape",
    'MiniNet': "Image shape has to be >200,200 for MobileNet. Do not forget that cropping does reduce image size",
}
