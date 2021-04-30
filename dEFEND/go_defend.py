# coding=utf-8
from __future__ import unicode_literals
from defend import Defend
import pandas as pd

#from bs4 import BeautifulSoup
import re
import numpy as np
#from sklearn.model_selection import train_test_split
from nltk import tokenize
#from keras.utils.np_utils import to_categorical
import operator
import pandas as pd

#!wget http://nlp.stanford.edu/data/glove.6B.zip
# unzip glove.6B.zip

#DATA_PATH = 'train_done.csv'
SAVED_MODEL_DIR = 'saved_models'
SAVED_MODEL_FILENAME = 'Defend.h5'
EMBEDDINGS_PATH = 'saved_models/glove.6B.100d.txt'


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


if __name__ == '__main__':
    # dataset used for training
    #platform = 'gossipcop'
    # platform = 'politifact'
    data_train = pd.read_csv('/home/alissa77/fake-news-detect/train_done.csv')
    data_test = pd.read_csv('/home/alissa77/fake-news-detect/test_done.csv')
    VALIDATION_SPLIT = 0.2
    '''    
    contents = []
    labels = []
    texts = []
    ids = []
    '''

    '''
    for idx in range(data_train.content.shape[0]):
        text = BeautifulSoup(data_train.content[idx], features="html5lib")
        text = clean_str(text.get_text().encode('ascii', 'ignore'))
        texts.append(text)
        sentences = tokenize.sent_tokenize(text)
        contents.append(sentences)
        ids.append(data_train.id[idx])

        labels.append(data_train.label[idx])

    labels = np.asarray(labels)
    labels = to_categorical(labels)

    # load user comments
    comments = []
    comments_text = []
    comments_train = pd.read_csv('data/' + platform + '_comment_no_ignore.tsv', sep='\t')
    print comments_train.shape

    content_ids = set(ids)

    for idx in range(comments_train.comment.shape[0]):
        if comments_train.id[idx] in  content_ids:
            com_text = BeautifulSoup(comments_train.comment[idx], features="html5lib")
            com_text = clean_str(com_text.get_text().encode('ascii', 'ignore'))
            tmp_comments = []
            for ct in com_text.split('::'):
                tmp_comments.append(ct)
            comments.append(tmp_comments)
            comments_text.extend(tmp_comments)
    '''

    x_train = data_train.claim
    x_val =  data_train.label
    y_train = data_test.claim
    y_val =  data_test.label
    #Qprint(x_train.shape,x_train[0].shape, 'X_train')
    #print(y_train.shape, 'y_train')

    # Train and save the model
    #SAVED_MODEL_FILENAME = platform + 'Defend.h5'

    #h = Defend(platform)
    h = Defend()
    #print(h.summary)
    #assert 6==5
    h.train(x_train, y_train, x_val, y_val,
            batch_size=20,
            epochs=30,
            embeddings_path='./glove.6B.100d.txt',
            saved_model_dir=SAVED_MODEL_DIR,
            saved_model_filename=SAVED_MODEL_FILENAME)

    h.load_weights(saved_model_dir = SAVED_MODEL_DIR, saved_model_filename = SAVED_MODEL_FILENAME)

    # Get the attention weights for sentences in the news contents as well as comments
    #activation_maps = h.activation_maps(x_val)
