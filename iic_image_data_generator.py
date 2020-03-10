import numpy as np
from tensorflow import keras
from typing import Tuple, List


class IIC_ImageDataGenerator(keras.utils.Sequence):
    """Image generator for IIC model training"""

    def __init__(self, parent_path: str, classes: List, heads:int, batch_size:int, image_size:Tuple=(200, 200)):
        # Keras generator
        self.generator = keras.preprocessing.image.ImageDataGenerator(rotation_range=15,
                                                                      #width_shift_range=0.1,
                                                                      #height_shift_range=0.1,
                                                                      #brightness_range=(
                                                                       #   0.5, 1.2),
                                                                      #shear_range=0.2,
                                                                      #zoom_range=0.2,
                                                                      #horizontal_flip=True,
                                                                      #vertical_flip=True,
                                                                      fill_mode='nearest', 
                                                                      rescale=1./255,)
                                                                      #preprocessing_function = self.preprocessing)
        self.reader = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255)
        self._path = parent_path
        self._classes = classes
        self._seed = np.random.choice(range(9999))
        self._image_size = image_size
        self._batch_size = batch_size
        self.heads = heads
        self._flow_args = dict(directory=self._path,
                         batch_size=self._batch_size,
                         classes=self._classes,
                         seed=self._seed,
                         color_mode='rgb',
                         target_size=self._image_size,
                         shuffle = True,)

        self._genX = self.reader.flow_from_directory(**self._flow_args)
        self._genX_t = self.generator.flow_from_directory(**self._flow_args)
        self._dataset_size = self._genX.n 

    def __len__(self):
        return self._genX.__len__()

    def __getitem__(self, index):
        """Getting items from the 2 generators and packing them"""
        X_batch, y1 = self._genX.__getitem__(index)
        Xt_batch, y2 = self._genX_t.__getitem__(index)

        x_train = np.concatenate([X_batch, Xt_batch], axis=0)
        y_train = np.concatenate([y1, y2], axis=0)
        y = []
        for _ in range(self.heads):
            y.append(y_train)
        return x_train, y

    def _preprocessing(self, image: np.ndarray):
        print("image shape ",image.shape)
        return image

    def get_input_shape(self):
        return (*self._image_size,3)

    def get_dataset_size(self):
        return self._dataset_size

    def get_batch_size(self):
        return self._batch_size


    