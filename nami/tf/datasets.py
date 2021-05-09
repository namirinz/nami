from glob import glob
import os
import tensorflow as tf
from typing import Union
import pandas as pd

class ImageFolder():    
    def __init__(self, paths: str, image_size: tuple, is_label:bool = True):
        """
            img_path/ <- * You pass this path *
            .../class1
            ....../image1.jpg
            ....../image2.jpg
            .../class2
            ....../image3.jpg
            ....../image4.jpg
        """
        self.filenames = glob(paths+'/*/*.jpg')
        self.image_size = image_size
        self.num_classes = 0
        self.is_label = is_label
        if is_label:
            self.LABEL = { os.path.basename(folder): i for i, folder in enumerate(glob(paths+'/*')) }
            self.num_classes = len(self.LABEL)
            self.labels = [self.LABEL[os.path.basename(os.path.dirname(filename))] for filename in self.filenames]
        print(f'found {len(self.filenames)} images with {self.num_classes} classes.')
    
    def __len__(self):
        return len(self.filenames)

    def setup_label(self, new_LABEL):
        self.LABEL = new_LABEL
        self.labels = [self.LABEL[os.path.basename(os.path.dirname(filename))] for filename in self.filenames]
    
    def _phrase_data(self, *args):
        img_ = tf.io.read_file(args[0])
        img_ = tf.image.decode_jpeg(img_, channels=3)
        img_ = tf.image.resize(img_, self.image_size) 

        if self.is_label:
            return img_, args[1]
        else: return img_

class ImageDataframe():
    def __init__(self, image_path: str, image_size: Union[tuple, list], df: pd.DataFrame, img_col: str, label_col: Union[str, list] = 'class', is_label: bool = True):
        """
            img_folder/ <- * You pass this path *
            .../image1.jpg
            .../image2.jpg
        """
        self.image_path = image_path
        self.filenames = (self.image_path + "/" + df[img_col]).values
        self.image_size = image_size
        self.is_label = is_label
        self.num_classes = 0
        if is_label:
            self.labels = df[label_col].values
            self.num_classes = len(df[label_col].unique())
        print(f'found {len(self.filenames)} images with {self.num_classes} classes.')
    def __len__(self):
        return len(self.filenames)
    
    def _phrase_data(self, *args):
        img_ = tf.io.read_file(args[0])
        img_ = tf.image.decode_jpeg(img_, channels=3)
        img_ = tf.image.resize(img_, self.image_size)

        if self.is_label:
            return img_, args[1]
        else: return img_