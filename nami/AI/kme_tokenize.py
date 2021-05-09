import tensorflow as tf
from tensorflow.keras.utils import get_file
import numpy as np
import json
from tqdm import tqdm
class Segmentation():
    def __init__(self):
        model_path = get_file(
            fname = 'my_model.hdf5',
            origin = "https://firebasestorage.googleapis.com/v0/b/ysc-kme-25095.appspot.com/o/best_model.hdf5?alt=media&token=bf2a8a81-33f2-4d11-84b6-522e01822cac"
        )

        char_path = get_file(
            fname = 'Char_Indices.json',
            origin = 'https://firebasestorage.googleapis.com/v0/b/ysc-kme-25095.appspot.com/o/vocab.json?alt=media&token=2ef47792-8ea2-49c9-9b6b-dd43e6786927'
        )
        
        with open(char_path, 'r') as char_f:
            self.CHAR_INDICES = json.load(char_f)
        
        self.model = tf.keras.models.load_model(model_path, compile=False)          
        
        self.look_back = 10
        
    def preprocessing_text(self, raw_text):
        """
        Take unseen (testing) text and encode it as number from CHAR_DICT.

        Parameters:
            raw_text (str): text to be encoded.
        
        Returns:
            numpy array: preprocessing text.
        """
        X = []
        data = [self.CHAR_INDICES['<pad>']] * self.look_back
        for char in raw_text:
            char = char if char in self.CHAR_INDICES else '<unk>'  # check char in dictionary
            data = data[1:] + [self.CHAR_INDICES[char]]  # X data
            X.append(data)
        return np.array(X)

    def predict(self, preprocessed_text):
        pred = self.model.predict(preprocessed_text)
        class_ = tf.argmax(pred, axis=-1).numpy()
        
        return class_

    def word_segmentation(self, text):
        """
        Segmenting text with AI.

        Parameters:
            text (str) : text to be segmented.
        
        Returns:
            numpy array : numpy array of segmented text.
            str : joined numpy array of segmented text.
        """
        preprocessed_text = self.preprocessing_text(text)
        class_ = self.predict(preprocessed_text)
        class_[0] = 1
        class_ = np.append(class_, 1)

        cut_indexs = [i for i, value in enumerate(class_) if value == 1]
        words = [text[cut_indexs[i]:cut_indexs[i+1]] for i in range(len(cut_indexs)-1)]
        
        join_word = '|'.join(words)
        
        return words, join_word

class Tokenizer:
    def __init__(self):
        self.word2index = {"<pad>": 0, "<start>": 1,"<end>": 2, "<unk>": 3}
        self.word2count = {"<pad>": 0, "<start>": 0,"<end>": 0, "<unk>": 0}
        self.index2word = {0: "<pad>", 1: "<start>", 2: "<end>", 3: "<unk>"}
        self.num_words = 4
        self.num_sentences = 0
        self.longest_sentences = 0
        self.kme_segment = Segmentation()
    
    def add_word(self, word):
        if word not in self.word2index:

            # First entry of word into vocabulary
            self.word2index[word] = self.num_words
            self.word2count[word] = 1
            self.index2word[self.num_words] = word
            self.num_words += 1
        else:

            # Word exists; increase word count
            self.word2count[word] += 1
    
    def fit_on_texts(self, sentences):
        """
        Generated bag of words from sentences.

        *No Return*

        Parameters:
            sentences (array of str) : text to fit on Tokenizer.
        """
        
        for sentence in tqdm(sentences):

            # Model predict 
            segmented_text, _ = self.kme_segment.word_segmentation(sentence)

            # Add <start> at start and <end> at end of each sentences
            segmented_text = np.concatenate((['<start>'], segmented_text, ['<end>']))
            
            if len(segmented_text) > self.longest_sentences:
                self.longest_sentences = len(segmented_text)

            for word in segmented_text:
                self.add_word(word)
        
    
    def text_to_sequences(self, sentences, method_pad='post', add_soe=False):
        """
        Preprocessing text into sequences (number) to be usable.

        Parameters:
            sentences (array of str) : To be segmented and preprocessed.
            method_pad ('pre' or 'post') : zero padding method.
                'pre': perform zero padding on front of word.
                'post: perform zero padding on end of word.
        
        Returns:
            array of int : Segmented and preprocessed text.
        """
        sequences_arr = []
        for sentence in tqdm(sentences):

            # Model predict 
            segmented_text, _ = self.kme_segment.word_segmentation(sentence)
            tokenize_text_arr = []
            
            for word in segmented_text:
                try:
                    tokenize_text_arr.append(self.word2index[word])
                except KeyError:

                    # use <unk> key for word that never met 
                    tokenize_text_arr.append(self.word2index['<unk>'])

            tokenize_text_arr = np.concatenate(([1], tokenize_text_arr, [2]))        
            
            sequences_arr.append(tokenize_text_arr)
        
        # Make zero padding by Maximum length of sentence
        sequences_padded = tf.keras.preprocessing.sequence.pad_sequences(
            sequences_arr, maxlen=self.longest_sentences, padding=method_pad
        )
        
        return sequences_padded
    
    def sequences_to_text(self, sequences):
        """
        Decoder sequences (number) into text (Organic Compound IUPAC name).

        Parameters:
            sequences (array of int) : To be decoded from index2word.

        Returns:
            array of str : Decoded sequences.
        """
        texts_arr = []
        for sequence in tqdm(sequences):
            index_arr = []
            for index in sequence:
                index_arr.append(self.index2word[index])
            texts_arr.append(index_arr)
        return np.array(texts_arr)