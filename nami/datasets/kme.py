import numpy as np
from tensorflow.keras.utils import get_file
from sklearn.model_selection import train_test_split
import json

def load_data(is_split=True, test_size=0.2):
    '''
    Get IUPAC organic compound datasets Image and Text

    Parameters:
        is_split (bool): split dataset into train and test dataset
        test_size (float): size of test dataset

    Returns:
        if is_split is True
        - 2 tuples of numpy array datasets image and text
            (image_train, text_train), (image_test, text_test)
        if is_split is False
        - Tuple of numpy array dataset image and text
            (image, text)
    '''

    dict_path = get_file(
        fname='kme_dict.json',
        origin='https://firebasestorage.googleapis.com/v0/b/ysc-kme-25095.appspot.com/o/Dict_segment.json?alt=media&token=297a3d1e-352a-4637-8917-5526df4ac574'
    )

    img_path = get_file(
        fname='kme_image.json',
        origin='https://firebasestorage.googleapis.com/v0/b/ysc-kme-25095.appspot.com/o/kme_dataset.npy?alt=media&token=531029d3-d563-4bef-b459-f02cbb74a76e'
    )    
    
    with open(dict_path, 'r') as f:
        dict = json.load(f)
    img_dataset = np.load(img_path)

    dict_arr = []
    for _, values in dict.items():
        dict_arr = np.concatenate((dict_arr, values), axis=0)
    
    if is_split:
        image_train, image_test, text_train, text_test = train_test_split(img_dataset, dict_arr, test_size=test_size, shuffle=True)
        return (image_train, text_train), (image_test, text_test)
    else:
        return (img_dataset, dict_arr)