import numpy as np
import pandas as pd
import jieba
from bert_serving.client import BertClient
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from operator import and_, or_
from functools import reduce
import os
import pickle
from scipy.spatial.distance import cosine
from itertools import combinations
import pickle
import baidu_spider
import random as rd

root_path = os.path.abspath(r'./')

qadata_path = os.path.join(root_path,'data','qadata.csv')
stop_words_path = os.path.join(root_path, 'data', 'stop_words.plk')
kmeans_path = os.path.join(root_path, 'kmeans_model', 'kmean.pkl')
# stop_words and removed words
stop_words = pickle.load(open(stop_words_path,'rb'))
remove_words = [
    '您好','你好','请问','如何','哪里','怎么','想'
]

# cut sentense into token strings seperated with space
def tokenlize(sentense):
    pattern = re.compile('[\w||\d]+')
    clear_sentence = ''.join(re.findall(pattern,sentense))
    return ' '.join(jieba.cut(clear_sentence))



# loading data
qadata = pd.read_csv(qadata_path)
qadata=qadata.dropna()
qadata['question_cut']=qadata.question.apply(tokenlize)

# get tfidf vector
# we need to define token_pattern to include single chinese character
tfidf =TfidfVectorizer(token_pattern='[\w\d]+') 
x=tfidf.fit_transform([w for w in qadata['question_cut']])
dense_x = x .toarray()
dense_x_transposed= dense_x.transpose()
word2id=tfidf.vocabulary_

# load kmeans model ,add lables for quesitons
# k_model = pickle.load(open(kmeans_path,'rb'))
# qadata['lable'] = pd.Series(k_model.labels_)


# filter for extracting key words 
def filter_keywords(inputs):
    words =list(jieba.cut(inputs))
    non_verbs = [ words[i] for i in range(len(words)) ]#if (pos[i] == 'n') or (pos[i]=='v')
    non_verbs = [ w for w in non_verbs if  (w not in stop_words) and (w not in remove_words) 
                and (w in word2id)]
#     print(words,'\r\n' ,pos,'\r\n',non_verbs)
#     tfidf.transform([' '.join(non_verbs)])
    return non_verbs

# find docs, boolen search
def get_merged_doc(doc_ids):
    i = len(doc_ids)
    result_sets =set()
    while i>0:
        sets_combis = list(combinations(doc_ids,i))
        for sets in sets_combis:
            set_and_set = set(reduce(and_, sets))
            if set_and_set: result_sets = or_(result_sets,set_and_set)
        if not result_sets: i= i-1 
        else: break
    if result_sets: return result_sets
    else: return None

# search for most similar questions, return answer
# sort_method =0 : sort with tfidf vec , =1: sort with bert sentence vec , =2: sort with tfidf and then vert vec
def search(inputs,sort_method,bc):
    words = filter_keywords(inputs)
    query_vec = tfidf.transform([' '.join(jieba.cut(inputs))]).toarray()[0]
    candidate_id = [word2id[w] for w in words if w in word2id]
    document_ids =[
        set(np.where(dense_x_transposed[_id])[0]) for _id in candidate_id
    ]
    if document_ids: 
#         merged_documents = list(reduce(and_, document_ids))
        merged_documents = list(get_merged_doc(document_ids))
        if merged_documents:
            if sort_method==2:
                sorted_merged_documents = sorted(merged_documents,key= lambda i: cosine(query_vec,dense_x[i]))[:10]#query_vec.dot(dense_x[i]),reverse=True
                query_vec_ber = bc.encode([inputs])[0]
                related_questions = qadata.question[sorted_merged_documents].values
                document_vec_bert = bc.encode(list(related_questions))
                sorted_documents = sorted(sorted_merged_documents,
                                          key = lambda i: cosine(query_vec_ber,document_vec_bert[sorted_merged_documents.index(i)]))
#                                           query_vec_ber.dot(document_vec_bert[sorted_merged_documents.index(i)]),reverse=True)
            elif sort_method==1:
                query_vec_ber = bc.encode([inputs])[0]
                related_questions = qadata.question[merged_documents].values
                document_vec_bert = bc.encode(list(related_questions))
                sorted_documents = sorted(merged_documents,
                                          key = lambda i: cosine(query_vec_ber, document_vec_bert[merged_documents.index(i)]))
            else:
                sorted_documents = sorted(merged_documents,key= lambda i: cosine(query_vec,dense_x[i]))
#                                           query_vec.dot(dense_x[i]),reverse=True)
            return sorted_documents
    else: return []

def talk(inputs,thresh_hold,bc):
#     inputs = '太阳到地球的距离'
    if ''.join(re.findall('\w+',inputs)) in ['你好','您好','你好啊','您好啊']:
        return rd.choice(
                [
                    ' 你好啊，很高兴见到你',
                    '你好，你好',
                    '好啊'
                ]
        )
    doc_indices = search(inputs,2,bc)
    if doc_indices:
        retrived_questins = qadata.question[doc_indices].values
        retrived_answers = qadata.answer[doc_indices].values
        bert_input_vec = bc.encode([inputs])[0]
        bert_topques_vec =  bc.encode([retrived_questins[0]])[0]
        similarity = cosine(bert_input_vec,bert_topques_vec)
        if similarity< thresh_hold:
            # print( inputs,'>>>>','\n---',   #retrived_questins[0],'>>>>',cosine(bert_input_vec,bert_topques_vec),
            #              retrived_answers[0]) #retrived_answers[0],
            return retrived_answers[0]
        else: 
            return baidu_spider.zhidao_search(inputs)
            # print(inputs,'\n---',baidu_spider.zhidao_search(inputs))
    else: 
        return baidu_spider.zhidao_search(inputs)
        # print(inputs,'\n---',baidu_spider.zhidao_search(inputs))

if __name__ == "__main__":
    # laoding bert service
    bc = BertClient()
    print(talk('你好',0.1,bc))
