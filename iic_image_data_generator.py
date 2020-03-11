import tensorflow as tf
import numpy as np
import cv2 as cv
from tensorflow import keras
from typing import Tuple, List


class IIC_ImageDataGenerator(keras.utils.Sequence):
    """
    @brief Image generator for IIC model training
    @remark ONLY ALLOWS IMAGES WITH SAME HEIGHT AND WIDTH
    """

    def __init__(self, parent_path: str, classes: List, heads: int, batch_size: int, crop_image: int = 0, image_size: Tuple = (200, 200)):
        """
        @param brief Keras image generator to load and to transform image data
        @param parent path Path including image data
        @param classes List of classes. For unlabeled data should be ['']
        @param heads number of heads
        @param batch_size batch size
        @param crop_image if value !=0, images are croped by given number of pixel
        @param image size: To which size should each image be resized
        """
        self.generator = keras.preprocessing.image.ImageDataGenerator(rotation_range=15,
                                                                      # width_shift_range=0.1,
                                                                      # height_shift_range=0.1,
                                                                      # brightness_range=(
                                                                      #   0.5, 1.2),
                                                                      # shear_range=0.2,
                                                                      # zoom_range=0.2,
                                                                      horizontal_flip=True,
                                                                      vertical_flip=True,
                                                                      fill_mode='nearest',
                                                                      rescale=1./255,)
        self.reader = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255,)
        self._path = parent_path
        self._classes = classes
        self._seed = np.random.choice(range(9999))
        self._image_size = image_size
        self._batch_size = batch_size
        self.heads = heads
        self._crop_image = crop_image
        self._flow_args = dict(directory=self._path,
                               batch_size=self._batch_size,
                               classes=self._classes,
                               seed=self._seed,
                               color_mode='rgb',
                               target_size=self._image_size,
                               shuffle=True,)
        self._genX = self.reader.flow_from_directory(**self._flow_args)
        self._genX_t = self.generator.flow_from_directory(**self._flow_args)
        self._dataset_size = self._genX.n

    def __len__(self):
        """
        @brief Has to be implemented
        """
        return self._genX.__len__()

    def __getitem__(self, index):
        """
        @brief Getting items from the 2 generators and packing them. 
                Apply image croping, because feature is not available in 
                keras ImageDataGenerator class
        """
        X_batch, y1 = self._genX.__getitem__(index)
        Xt_batch, y2 = self._genX_t.__getitem__(index)

        # crop images
        X_batch, Xt_batch = self._crop_images(X_batch, Xt_batch)

        x_train = np.concatenate([X_batch, Xt_batch], axis=0)
        y_train = np.concatenate([y1, y2], axis=0)
        y = []
        for _ in range(self.heads):
            y.append(y_train)
        return x_train, y

    def _crop_images(self, X_batch, Xt_batch):
        """
        @brief: crops batches X and Xt. X is central corped. 
                Xt is randomly croped
        """
        if self._crop_image == 0:
            return X_batch, Xt_batch

        new_shape = (self._batch_size, *tuple((i-self._crop_image)
                                              for i in self._image_size), 3)

        X_batch_new = np.zeros(new_shape)
        Xt_batch_new = np.zeros(new_shape)
        for i in range(len(X_batch)):
            X_batch_new[i] = self._central_croping(X_batch[i])
            Xt_batch_new[i] = self._random_croping(Xt_batch[i])

        return X_batch_new, Xt_batch_new

    def _central_croping(self, image: np.ndarray) -> np.ndarray:
        """
        @brief uses tensorflow image function to crop image
        """
        crop_ratio = self._crop_image / float(self._image_size[0])
        i = tf.image.central_crop(image, 1-crop_ratio)
        return i

    def _random_croping(self, image: np.ndarray) -> np.ndarray:
        """
        @brief uses tensorflow image function to randomly crop the input image
        """
        size = tuple((i - self._crop_image) for i in self._image_size)
        if np.random.randint(0, 2):
            i = tf.image.random_crop(image, (*size, 3)).numpy()
            return i
        else:
            i = cv.resize(image, size)
            return i

    def get_input_shape(self) -> Tuple:
        """
        @brief Getter method to obtain input shape of generator
        """
        return (*self._image_size, 3)

    def get_dataset_size(self) -> int:
        """
        @brief Getter method to obtain size of the whole dataset
        """
        return self._dataset_size

    def get_batch_size(self) -> int:
        """
        @brief returns the batch size
        """
        return self._batch_size
