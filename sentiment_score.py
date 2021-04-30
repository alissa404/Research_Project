import csv
import pandas as pd
import pickle

def sentiment(path):
    input= pd.read_csv(path)
    # pandas to dict
    lyrics = input.WS.to_dict()

    # D: degree
    # Ew: emotion word
    # N: negative adv
    # valence: negative to positve

    # 程度副詞 #md
    degree = {'最':0.9, '最為':0.9, '極':0.9, '極為':0.9, '極其':0.9, '極度':0.9, '極端':0.9, '至':0.9, '至為':0.9,\
            '頂':0.9, '過':0.9, '過於':0.9, '過分':0.9, '分外':0.9, '萬分':0.9,\
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
        score_degree = 0
        score_negative = 0
        len_cvaw = 1 

        for i in range(2, len(lyrics[ID])): # lyric[i] = a word in CVAW dictionary
            lyric = lyrics[ID]
            temp_v = 0  # vep
            temp_a = 0  # aep

            if  lyrics[ID] in list(degree):
                score_degree += degree[ lyrics[ID]]
            if  lyrics[ID] in list(negative):
                score_negative += negative[ lyrics[ID]]
            
            if lyric[i] in arousal:
                len_cvaw += 1
                this_v = valence[lyric[i]]  # Vew
                this_a = arousal[lyric[i]]  # Aew

                #input裡面只要包含 degree{}, nagetive{} 的字，就會得到分數，不管有沒有emotion word
                # ex:
                # ['比特幣', '絕對', '是', '穩賺', '不賠']
        
                # if lyric[i] in degree:
                #     temp_v = degree[lyric[i]]
                #     temp_a = degree[lyric[i]]
                
                # elif lyric[i] in negative:
                #     temp_v = negative[lyric[i]]
                #     temp_a = negative[lyric[i]]


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

        # 排除情況（第四象限
        valence_output = score_v/len_cvaw
        if (score_a < 0 and valence_output > 0):
            score_a = 0

        score_a += score_degree
        
        
        print(lyrics[ID])
        #print ("Mean_valence:", valence_output)
        print ("Mean_arousal:", score_a)
        print("Degree_score:", score_degree)  # error : always zero

        input["sentiment"] = score_a    #append  "score_a "  in  "input" dataframe
        