import os
import pickle
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import torch
import sys

from bert.extract_feature import BertVector
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.optimizers import SGD, Adagrad
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import Concatenate, dot, Layer
from tensorflow.keras import backend as K
from tensorflow.keras import initializers
from tensorflow import keras
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Embedding, Input, Lambda
from sklearn.metrics import classification_report, confusion_matrix
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
from tensorflow.keras.layers import MaxPooling1D, GlobalMaxPooling1D, AveragePooling1D, LayerNormalization,GlobalAveragePooling1D, TimeDistributed


from sentiment_score import sentiment



def cat_replace(df):
    df.category.replace("生活","0",inplace=True)
    df.category.replace("健康","1",inplace=True)
    df.category.replace("娛樂","0",inplace=True)
    df.category.replace("體育","2",inplace=True)
    df.category.replace("政治","3",inplace=True)
    df.category.replace("財經","4",inplace=True)
    df.category.replace("財金","4",inplace=True)
    df.category.replace("社會","0",inplace=True)
    df.category.replace("國際外交","3",inplace=True)
    return df


def get_pos(df):
    claims = df['claim']
    sentence_list = []
    for claim in claims:
        sentence_list.append(claim)

    word_sentence_list = ws(
    sentence_list,
    # sentence_segmentation = True, # To consider delimiters
    # segment_delimiter_set = {",", "。", ":", "?", "!", ";"}), # This is the defualt set of delimiters
    recommend_dictionary = dictionary, # words in this dictionary are encouraged
    # coerce_dictionary = dictionary2, # words in this dictionary are forced
    )
    pos_sentence_list = pos(word_sentence_list)
    df['pos'] = pos_sentence_list
    df['ws'] = word_sentence_list
    return df


#stopwords
def stopwords(df):
    stop_words = open('stops.txt', 'r')
    lines = stop_words.readlines()
    stop_words_dict = {}
    for line in lines:
        line = line.replace('\n','')
        stop_words_dict[line] = 1
    return df

'''
# padding 
def padding(df):
    for idx in range(0, df.shape[0]):
    sys.stdout.write('\r'+ "Implementing BERT vectoring {}% \n".format(round(100 * idx/df.shape[0], 2)))
    title_list = []
    for word in df[target_feature].iloc[idx]:
            vec = bv.encode([word])
            title_list.append(vec)
            df[target_feature + "_vector"].iloc[idx] = title_list
    return df
'''

# POS to Index       
def pos2idx():
    pos_index = {}
    idx = 1
    for index, rows in df_train.POS.iterrows() :
        for pos in rows['POS'] :
            if pos not in pos_index :
                pos_index[pos] = idx
                idx += 1
                    
    for index, rows in df_test.POS.iterrows() :
        for pos in rows['POS'] :
            if pos not in pos_index :
                pos_index[pos] = idx
                idx += 1


class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)

        self.outputt = x + positions
        return x + positions
    def compute_output_shape(self, input_shape):

        return self.outputt.shape[0],self.outputt.shape[1],self.outputt.shape[2]

# encoder
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim),])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        self.outputt = self.layernorm2(out1 + ffn_output)
        return self.layernorm2(out1 + ffn_output)
    
    def compute_output_shape(self, input_shape):

        return self.outputt.shape[0],self.outputt.shape[1],self.outputt.shape[2]


# multihead self attention
class MultiHeadSelfAttention(Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        self.outputt = output
        return output

#HAN
class HierarchicalAttentionNetwork(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(HierarchicalAttentionNetwork, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim,)))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.train_weights = [self.W, self.b, self.u]
        super(HierarchicalAttentionNetwork, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))

        ait = K.exp(K.squeeze(K.dot(uit, self.u), -1))

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        weighted_input = x * K.expand_dims(ait)
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):

        return input_shape[0], input_shape[-1]


#model
class Classifier(nn.Module):
    def __init__(self, config):
        cat_input = Input(shape = ((2*max_head_len )), dtype = 'int32', name = 'sentEncoder_input')
        embedding = Embedding(len(word_index)+1, 300,weights=[embedding_matrix],trainable = False)
        pos_embedding = TokenAndPositionEmbedding(251, len(pos_index)+1, 8)

        pos_encoder = TransformerBlock(8, 1, 8)

        head_input = Lambda(lambda x: x[:, 0:(max_head_len)], output_shape=(max_head_len))(cat_input)
        head_pos_input = Lambda(lambda x: x[:, (max_head_len):(2*max_head_len )])(cat_input)

        embedded_sequences = embedding(head_input)
        #print(embedded_sequences.shape)
        lstm_word = Bidirectional(GRU(128, return_sequences=True))(embedded_sequences)
        head_body_word_att_out = HierarchicalAttentionNetwork(100)(lstm_word)

        embedded_head_pos = pos_embedding(head_pos_input)
        head_pos_out = pos_encoder(embedded_head_pos)
        #print(head_pos_out.shape)
        head_pos_out= Dropout(0.9)(head_pos_out)
        head_pos_out =  GlobalAveragePooling1D()(head_pos_out )
        head_pos_out  = Dense(10, activation="relu")(head_pos_out  )
        head_pos_out= Dropout(0.9)(head_pos_out)

        sub_out = Concatenate(axis = -1)([head_body_word_att_out,head_pos_out])
        head_sent_encoder = Model(cat_input,sub_out)

        head = Input(shape=(100,30), dtype = 'int32')
        head_pos = Input(shape=(100,30), dtype = 'int32')
        concat = Concatenate(axis = -1)([head, head_pos, sentiment])

        review_encoder = TimeDistributed(head_sent_encoder)(concat)
        #print(review_encoder.shape)

        claims = Bidirectional(GRU(128, return_sequences=True))(review_encoder)
        claims = Dropout(0.9)(claims)
        claims_att = HierarchicalAttentionNetwork(100)(claims)
        claims_att = Dense(200,activation='relu')(claims_att)
        preds = Dense(3, activation='softmax')(claims_att)
        model = Model([claims,claims_pos,score_s], preds)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


if __name__ == '__main__':
    word_to_weight = {
    "武漢肺炎": 10,  "武漢病毒": 10, "川普": 10,
    "川普": 10,"拜登": 10,"蔡總統": 10,"馬總統": 10,
    "新型肺炎": 10,"新型病毒": 10,"新冠病毒": 10,"新冠肺炎": 10,
    "公有": 2,"柯P": 10,"世界衛生組織": 10,  "川普": 1,
    "中央疫情指揮中心": 10, "中央流行疫情指揮中心": 10,"加利": 1,
    "變種病毒": 100, "輝瑞疫苗": 100, "蠟筆小新": 100,
    "新型流感": 1, "冠狀病毒": 1,
    "武肺": 1,   "指揮中心": 1,
    "口罩": 1,"酷碰券": 100,"蔡英文": 100,"蔡政府": 100,
    }
    dictionary = construct_dictionary(word_to_weight)

    # load model
    bv = BertVector()
    ws = WS('data')
    pos = POS('data')
    ner = NER('data')

    #sentiment_score
    score_s = sentiment("train_done.csv")

    # max_len
    max_head_len = 150 

    # load data
    df_train = cat_replace(pd.read_csv('train_done.csv')).drop_duplicates(subset=['claim']).dropna()
    df_test  = cat_replace(pd.read_csv('test_done.csv')).drop_duplicates(subset=['claim']).dropna()




'''
    class SequenceCriteria(nn.Module):
        def __init__(self, class_weight):
            super(SequenceCriteria, self).__init__()
            self.criteria = nn.CrossEntropyLoss(weight=class_weight)

        def forward(self, inputs, targets):
            # This is BxT, which is what we want!
            loss = self.criteria(inputs, targets)
            return loss


    def _linear(in_sz, out_sz, unif):
        
        l = nn.Linear(in_sz, out_sz)
        weight_init.xavier_uniform(l.weight.data)
        return l


    def _append2seq(seq, modules):
        for module_ in modules:
            seq.add_module(str(module_), module_)


    def binary_cross_entropy(x, y, smoothing=0., epsilon=1e-12):
        y = y.float()
        if smoothing > 0:
            smoothing *= 2
            y = y * (1 - smoothing) + 0.5 * smoothing
        return -torch.mean(
            torch.log(x + epsilon) * y + torch.log(1.0 - x + epsilon) * (1 - y))

    # 用bi-gru
    class Classifier(nn.Module):
        def __init__(self, config):
            super(Classifier, self).__init__()
            self.config = config
            seq_in_size = config.d_hidden
            if config.brnn:
                seq_in_size *= 2
            if self.config.use_addn or True:
                layers1 = [_linear(config.addn_dim-5, 64, config.init_scalar),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        _linear(64, 256, config.init_scalar),
                        nn.ReLU()]
                layers2 = [_linear(config.addn_dim-14, 64, config.init_scalar),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        _linear(64, 256, config.init_scalar),
                        nn.ReLU()]
                layers3 = [_linear(1024, 1024, config.init_scalar),
                        nn.ReLU(),
                        nn.Dropout(0.3)]     
                layers4 = [_linear(1024+256, 1024, config.init_scalar),
                        nn.ReLU(),
                        nn.Dropout(0.3)]
                layers5 = [_linear(256, 256, config.init_scalar),
                        nn.ReLU()]
                layers6 = [_linear(300, 300, config.init_scalar),
                        nn.ReLU()]
                
                self.addn_model1 = nn.Sequential(*layers1)
                self.addn_model2 = nn.Sequential(*layers2)
                self.addn_model3 = nn.Sequential(*layers3)
                self.addn_model4 = nn.Sequential(*layers4)
                self.addn_model5 = nn.Sequential(*layers5)
                self.ext_model1 = nn.Sequential(*layers6)
                
                seq_in_size += 256+300
                
            if config.down_projection:
                self.down_projection = _linear(seq_in_size,
                                            config.d_down_proj,
                                            config.init_scalar)
                self.act = nn.Sigmoid()
                seq_in_size = config.d_down_proj
                
            self.clf = _linear(seq_in_size,
                            config.num_classes,
                            config.init_scalar)
           
        def attention_net(self, a, b):
            self.weights1 = torch.bmm(a,
                                    b.reshape(1, -1,
                                                self.config.d_hidden*2).squeeze(0).unsqueeze(2)).squeeze(2)
            self.weights2 = F.softmax(self.weights1).unsqueeze(2)
            return torch.bmm(a.transpose(1,2), self.weights2).squeeze(2)


    class Training(object):
        def __init__(self, config, logger=None):
            if logger is None:
                logger = logging.getLogger('logger')
                logger.setLevel(logging.DEBUG)
                logging.basicConfig(format='%(message)s', level=logging.DEBUG)

            self.logger = logger
            self.config = config
            self.classes = list(config.id2label.keys())
            self.num_classes = config.num_classes

            #self.embedder = Embedder(self.config)
            #self.encoder = LSTMEncoder(self.config)
            self.clf = Classifier(self.config)
            self.clf_loss = SequenceCriteria(class_weight=None)
            if self.config.lambda_ae > 0: self.ae = AEModel(self.config)

            self.writer = SummaryWriter(log_dir="TFBoardSummary")
            self.global_steps = 0
            self.enc_clf_opt = Adam(self._get_trainabe_modules(),
                                    lr=self.config.lr,
                                    betas=(config.beta1,
                                        config.beta2),
                                    weight_decay=config.weight_decay,
                                    eps=config.eps)

            if config.scheduler == "ReduceLROnPlateau":
                self.scheduler = lr_scheduler.ReduceLROnPlateau(self.enc_clf_opt,
                                                                mode='max',
                                                                factor=config.lr_decay,
                                                                patience=config.patience,
                                                                verbose=True)
            elif config.scheduler == "ExponentialLR":
                self.scheduler = lr_scheduler.ExponentialLR(self.enc_clf_opt,
                                                            gamma=config.gamma)

            self._init_or_load_model()
            if config.multi_gpu:
                self.embedder.cuda()
                self.encoder.cuda()
                self.clf.cuda()
                self.clf_loss.cuda()
                if self.config.lambda_ae > 0: self.ae.cuda()

            self.ema_embedder = ExponentialMovingAverage(decay=0.999)
            self.ema_embedder.register(self.embedder.state_dict())
            self.ema_encoder = ExponentialMovingAverage(decay=0.999)
            self.ema_encoder.register(self.encoder.state_dict())
            self.ema_clf = ExponentialMovingAverage(decay=0.999)
            self.ema_clf.register(self.clf.state_dict())

            self.time_s = time()

        def _get_trainabe_modules(self):
            param_list = list(self.embedder.parameters()) + \
                        list(self.encoder.parameters()) + \
                        list(self.clf.parameters())
            if self.config.lambda_ae > 0:
                param_list += list(self.ae.parameters())
            return param_list
'''

