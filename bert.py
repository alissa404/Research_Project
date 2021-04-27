import os
import pickle
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import csv
import torch

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


def load_data():
    target_feature = 'claim'
    train_df = cat_replace(pd.read_pickle('data_vector.pkl')).drop_duplicates(subset=['claim']).dropna()
    test_df = cat_replace(pd.read_csv('test.csv')).drop_duplicates(subset=['claim']).dropna()
    return test_df


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

# lyrics = claims
class sentiment_analyzer(lyrics):
    def analyzer(lyrics):  
'''        
#example
lyrics = {          
        3: ['比特幣','絕對','是','穩賺','不賠'],\
        4:['我','好不','想','上班'],\
        5:['不是','打','過','疫苗','就','不會','重症','死亡'],\
        6:['兩位','醫師','都','早已','打','過','2','劑','輝瑞疫苗'],\
        7:['現在','變種病毒','株','已','不是','武漢病毒','株']   
        }
'''
# 程度副詞 #md
degree = {'最':0.9, '最為':0.9, '極':0.9, '極為':0.9, '極其':0.9, '極度':0.9, '極端':0.9, '至':0.9, '至為':0.9,\
           '頂':0.9, '過':0.9, '過於':0.9, '過分':0.9, '分外':0.9, '萬分':0.9, '絕對':0.9, '一定':0.9,'肯定':0.9,\
           '更':0.7, '更加':0.7, '更為':0.7, '更其':0.7, '越':0.7, '越發':0.7, '備加':0.7, '愈加':0.7, '愈':0.7,\
           '愈發':0.7, '愈為':0.7, '愈益':0.7, '越加':0.7, '格':0.7, '益發':0.7, '還':0.7, '很':0.7, '太':0.7,'都':0.7,\
           '挺':0.7, '怪':0.7, '老':0.7, '非常':0.7, '特別':0.7, '相當':0.7, '十分':0.7, '好':0.7, '好不':-1,
           '甚':0.7, '甚為':0.7, '頗':0.7, '頗為':0.7, '異常':0.7, '深為':0.7, '滿':0.7, '蠻':0.7, '夠':0.7, '多':0.7,\
           '多麼':0.7, '舒':0.7, '特':0.7, '大':0.7, '大為':0.7, '何等':0.7, '何其':0.7, '尤其':0.7, '無比':0.7, '尤為':0.7, '不勝':0.7,\
           '較':0.5, '比較':0.5, '較比':0.5, '較為':0.5, '還':0.5, '不大':0.5, '不太':0.5, '不很':0.5, '不甚':0.5,'已':0.5,'早已':0.6,'不是':0.5,\
           '稍':-0.5, '稍稍':-0.5, '稍為':-0.5, '稍微':-0.5, '稍許':-0.5, '略':-0.5, '略略':-0.5, '略為':-0.5,\
           '些微':-0.5, '多少':-0.5, '有點':-0.5, '有點兒':-0.5, '有些':-0.5}

# 否定副詞 #mn
negative = {'白':-0.8, '白白':-0.8, '甭':-0.8, '別':-0.8, '不':-0.8, '不必':-0.8, '不曾':-0.8, '不要':-0.8,\
          '不用':-0.8, '非':-0.8, '幹':-0.8, '何必':-0.8, '何曾':-0.8, '何嘗':-0.8, '何須':-0.8,\
          '空':-0.8, '沒':-0.8, '沒有':-0.8, '莫':-0.8, '徒':-0.8, '徒然':-0.8, '忹':-0.8,\
          '未':-0.8, '未曾':-0.8, '未嘗':-0.8, '無須':-0.8, '無須乎':-0.8, '無需':-0.8, '毋須':-0.8,\
          '毋庸':-0.8, '無庸':-0.8, '勿':-0.8, '瞎':-0.8, '休':-0.8, '虛':-0.8}


def b_edge(value):
    if value<0:
        return -4
    else:
        return 4

valence = {}
arousal = {}
with open('cvaw4.csv', newline='') as csvfile:
    rows = csv.DictReader(csvfile)
    for row in rows:
        valence[row['Word']] = (row['Valence_Mean'])
        arousal[row['Word']] = (row['Arousal_Mean'])
        
valence = {x: round(float(valence[x])-5, 2) for x in valence}
arousal = {x: round(float(arousal[x])-5, 2) for x in arousal}

for ID in lyrics:
    score_v = 0
    score_a = 0
    len_cvaw = 1 

    for i in range(2, len(lyrics[ID])): # lyric[i] = a word in CVAW dictionary
        lyric = lyrics[ID]
        temp_v = 0  # vep
        temp_a = 0  # aep

        if lyric[i] in arousal:
            len_cvaw += 1
            this_v = valence[lyric[i]]  # Vew
            this_a = arousal[lyric[i]]  # Aew
  




            #input裡面只要包含 degree{}, nagetive{} 的字，就會得到分數，不管有沒有emotion word
            '''
            ex:
            ['比特幣', '絕對', '是', '穩賺', '不賠']
            Mean_arousal: 0.9   #'絕對'出現在degree{'絕對':0.9...}         
            '''
            if lyric[i] in degree:
                temp_v = degree[lyric[i]]
                temp_a = degree[lyric[i]]
            
            elif lyric[i] in negative:
                temp_v = negative[lyric[i]]
                temp_a = negative[lyric[i]]
             
            # 以上這段code沒有成功實行




            if lyric[i-2] in negative:
                if lyric[i-1] in negative:
                    print ("N + N + EW:", lyric[i-2:i+1:])
                    param = negative[ lyric[i-2] ] * negative[ lyric[i-1] ]
                    temp_v = param * this_v
                    temp_a = param * this_a
                    
                elif lyric[i-1] in degree:
                    print ("N + D + EW:", lyric[i-2:i+1:])
                    param = degree[ lyric[i-1] ] - (1 + negative[ lyric[i-2] ])
                    temp_v = this_v + (b_edge(this_v) - this_v) * param
                    temp_a = this_a + (b_edge(this_a) - this_a) * param
                    
            elif lyric[i-2] in degree:
                if lyric[i-1] in negative:
                    print ("D + N + EW:", lyric[i-2:i+1:])
                    mn = negative[ lyric[i-1] ]
                    md = degree[ lyric[i-2] ]
                    param_v = mn * this_v
                    param_a = mn * this_a
                    temp_v = param_v + (b_edge(this_v) - param_v) * md
                    temp_a = param_a + (b_edge(this_a) - param_a) * md
                    
                elif lyric[i-1] in degree:
                    print ("D + D + EW:", lyric[i-2:i+1:])
                    md_1 = degree[ lyric[i-1] ]
                    md_2 = degree[ lyric[i-2] ]
                    param_v = (b_edge(this_v) - this_v) * md_1
                    param_a = (b_edge(this_a) - this_a) * md_1
                    temp_v = this_v + param_v + (1 - (this_v + param_v)) * md_2
                    temp_a = this_a + param_a + (1 - (this_a + param_a)) * md_2
                
            elif lyric[i-1] in negative:
                print ("N + EW:", lyric[i-1:i+1:])
                temp_v = negative[ lyric[i-1] ] * valence[lyric[i]]
                temp_a = negative[ lyric[i-1] ] * arousal[lyric[i]]
                
            elif lyric[i-1] in degree:
                print ("D + EW:", lyric[i-1:i+1:])
                temp_v = this_v + (b_edge(this_v) - this_v) * degree[ lyric[i-1] ]
                temp_a = this_a + (b_edge(this_a) - this_a) * degree[ lyric[i-1] ]
                
            else:
                print ("EW:", lyric[i])   
                temp_v = valence[lyric[i]]
                temp_a = arousal[lyric[i]]

        score_v += temp_v
        score_a += temp_a

    valence_output = score_v/len_cvaw

    # 排除情況（第四象限
    if (score_a < 0 and valence_output > 0):
        score_a = 0

    print(lyrics[ID])
    #print ("Mean_valence:", valence_output)
    print ("Mean_arousal:", score_a)



    





    
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
        #cat_input = Input(shape = ((2*max_head_len + 2*max_body_sent_len)), dtype = 'int32', name = 'sentEncoder_input')
        #shared_embedding = Embedding(len(word_index)+1, 300,weights=[embedding_matrix],trainable = False)
        pos_embedding = TokenAndPositionEmbedding(150, len(pos_index)+1, 8)
        pos_encoder = TransformerBlock(8, 1, 8)

        head_body_input = Lambda(lambda x: x[:, 0:(max_head_len+max_body_sent_len)], output_shape=(max_head_len+max_body_sent_len))(cat_input)
        head_pos_input = Lambda(lambda x: x[:, (max_head_len+max_body_sent_len):(2*max_head_len + max_body_sent_len)])(cat_input)
        body_pos_input = Lambda(lambda x: x[:, (2*max_head_len + max_body_sent_len):(2*max_head_len + 2*max_body_sent_len)])(cat_input)

        embedded_sequences = shared_embedding(head_body_input)
        #print(embedded_sequences.shape)
        lstm_word = Bidirectional(GRU(128, return_sequences=True))(embedded_sequences)
        head_body_word_att_out = HierarchicalAttentionNetwork(100)(lstm_word)

        embedded_body_pos = pos_embedding(body_pos_input)
        body_pos_out = pos_encoder(embedded_body_pos)
        body_pos_out = Dropout(0.9)(body_pos_out)
        body_pos_out =  GlobalAveragePooling1D()(body_pos_out )
        body_pos_out = Dense(10, activation="relu")(body_pos_out  )
        body_pos_out= Dropout(0.9)(body_pos_out)

        sub_out = Concatenate(axis = -1)([head_body_word_att_out,head_pos_out,body_pos_out])
        head_sent_encoder = Model(cat_input,sub_out)

        head = Input(shape=(100,30), dtype = 'int32')
        body = Input(shape=(100,251), dtype = 'int32')
        head_pos = Input(shape=(100,30), dtype = 'int32')
        body_pos = Input(shape = (100,251), dtype = 'int32')
        concat = Concatenate(axis = -1)([head, body, head_pos,body_pos])

        #review_encoder = TimeDistributed(head_sent_encoder)(concat)
        #print(review_encoder.shape)

        claims = Bidirectional(GRU(128, return_sequences=True))(review_encoder)
        claims = Dropout(0.9)(claims)
        claims_att = HierarchicalAttentionNetwork(100)(claims)
        claims_att = Dense(200,activation='relu')(claims_att)
        preds = Dense(3, activation='softmax')(claims_att)
        model = Model([claims,claims_pos], preds)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])


if __name__ == '__main__':
    word_to_weight = {
    "武漢肺炎": 10,
    "武漢病毒": 10,
    "川普": 10,
    "川普": 10,"拜登": 10,"蔡總統": 10,"馬總統": 10,
    "新型肺炎": 10,"新型病毒": 10,"新冠病毒": 10,"新冠肺炎": 10,
    "公有": 2,"柯P": 10,"世界衛生組織": 10,
    }
    dictionary = construct_dictionary(word_to_weight)

    # load model
    bv = BertVector()
    ws = WS('data')
    pos = POS('data')
    ner = NER('data')

    test_df = load_data()
    test_df = get_pos(test_df)  
    ws = test_df['ws']

    # update
    ner = test_df['ner']


    #sentiment_analyzer
    for s in enumerate(ws):
        lyrics = s
        '''
        舉例：
        lyrics = {
        1: ['我','不','想','上班'],\
        2:['我','好不','想','上班'],\
        3:['claim_text']...
        }
        '''
        sentiment_score = sentiment_analyzer(lyrics)

    # max_len
    max_claim_len = 150  #config



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

