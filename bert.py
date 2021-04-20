import os
import pickle
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import csv

from tensorflow import keras
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from ckiptagger import data_utils, construct_dictionary, WS, POS, NER
from bert.extract_feature import BertVector


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
    # recommend_dictionary = dictionary1, # words in this dictionary are encouraged
    # coerce_dictionary = dictionary2, # words in this dictionary are forced
    )
    pos_sentence_list = pos(word_sentence_list)
    df['pos'] = pos_sentence_list
    df['ws'] = word_sentence_list
    return df




#new_functions
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


class sentiment_analyzer(lyrics):
    def analyzer(lyrics):  

#example
lyrics = {          
        3: ['比特幣','絕對','是','穩賺','不賠'],\
        4:['我','好不','想','上班'],\
        #5:['不是','打','過','疫苗','就','不會','重症','死亡'],\
        #6:['兩位','醫師','都','早已','打','過','2','劑','輝瑞疫苗'],\
        #7:['現在','變種病毒','株','已','不是','武漢病毒','株']   
        }

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

    # 需排除第四象限
    if (score_a < 0 and valence_output > 0):
        score_a = 0

    print(lyrics[ID])
    #print ("Mean_valence:", valence_output)
    print ("Mean_arousal:", score_a)





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


'''
update below
'''
    #stopwords()
    ws_ = stopwords(ner)

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

'''
    # 載入一個可以做中文多分類任務的模型
    from transformers import BertForSequenceClassification
    import torch
    from transformers import BertTokenizer
    from IPython.display import clear_output

    PRETRAINED_MODEL_NAME = "bert-base-chinese"
    NUM_LABELS = 2 #output_class_number

    model = BertForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)

    clear_output()

    # high-level 顯示此模型裡的 modules
    print("""
    name            module
    ----------------------""")
    for name, module in model.named_children():
        if name == "bert":
            for n, _ in module.named_children():
                print(f"{name}:{n}")
        else:
            print("{:15} {}".format(name, module))
'''

def evaluate():