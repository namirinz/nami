# nami
nami is python package
# Installation
`pip install --upgrade nami`
# Features (nami-1.2.0.1)
### get_dataset

 ```python
from nami.datasets.ImageNet import get_dataset
dataset = get_dataset(noun='STR', dimension=(INT,INT), max=INT, timeout=FLOAT, save_to='STR')
```
> get 'INT*INT dimenstion' of 'noun' image dataset from ImageNet.  
> timeout - [from 0.1 - 1.0] maximum time request for each image URL.
> max - number of images dataset.
> save_to - save the dataset by '.npy' format.

### load_dataset (KME)
```python
from nami.datasets.kme import load_data
(X_train, y_train), (X_test, y_test) = load_data(is_split=True. test_size=0.2)
(X, y) = load_data(is_split=False)
```

### Tokenizer class
 ```python
from nami.AI.kme_tokenize import Tokenizer
tokenizer = Tokenizer()

text_arr = ['methyl methanoate', 'ethane', '(hydroxymethylamino)oxy-methoxymethanol']
 
```python
tokenizer.fit_on_text(text_arr)
```
> fit_on_text(sentences=)
> <br>sentences: take **array of string** to make bag of words (word2index & index2word)
```python
train_seq = tokenizer.text_to_sequences(text_arr, method_pad='pre')
```
>text_to_sequences(sentences= , method_pad='post')
> <br>sentences: take **array of string** to preprocessing text to numeric
> <br>method_pad: **('pre', 'post')** make zero padding
```
train_seq

[[ 0  0  0  0  0  0  0  4  5  6  4  7  8]
 [ 0  0  0  0  0  0  0  0  0  0  0  9 10]
 [11 12  4  5 13 14 15 16  4 15  4  7 17]]
```

```python
test_arr = ['2-(4-methoxyphenyl)-2-oxoacetic acid']

test_seq = tokenizer.text_to_sequences(test_arr)
# [[11, 14, 18, 13, 14, 4, 22, 3, 5, 21, 14, 11, 14, 3, 3, 3]]

test_text = tokenizer.sequences_to_text(test_seq)
# [['2', '-', '(', '4', '-', 'meth', 'oxy', '<unk>', 'yl', ')', '-', '2', '-', '<unk>', '<unk>', '<unk>']]
```
