# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 11:45:06 2018

@author: 45016577
"""

# -*- coding: utf-8 -*-

"""
Created on Fri Aug 17 12:20:19 2018
Generate user and item vectors by learning their profiles, item details, and reviews
@author: Feng Zhu
Function Doc2vec
Url:"https://radimrehurek.com/gensim/models/doc2vec.html#gensim.models.doc2vec.Doc2Vec"
"""
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim import corpora
from gensim.models import Phrases
from nltk.corpus import stopwords
import nltk
import os
import string
from stanfordcorenlp import StanfordCoreNLP
import pymysql.cursors

def is_chinese(uchar):
    """is this a chinese word?"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False 

def is_number(uchar):
    """is this unicode a number?"""
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False

def is_alphabet(uchar):
    """is this unicode an English word?"""
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a') or uchar == " ":
        return True
    else:
        return False
    
def format_str(content,lag):
    content_str = ''
    if lag==0: #English
       for i in content:
           if is_alphabet(i):
               content_str = content_str+i
    if lag==1: #Chinese
        for i in content:
            if is_chinese(i):
                content_str = content_str+i
    if lag==2: #Number
        for i in content:
            if is_number(i):
                content_str = content_str+i        
    return content_str
    
#get nlp
nlp = StanfordCoreNLP('C:/Users/45016577/Downloads/stanford-corenlp-full-2018-02-27/')

#clean stops
nltk.download('stopwords')
#Connect to mysql 
# Configure a connector
connect = pymysql.Connect(
    host='localhost',
    port=3306,
    user='root',
    passwd='123456',
    db='epinions_experiment',
    charset='utf8'
)

#All documents (for users)
item_review=[]

# For items

cursor1=connect.cursor()
sql ="SELECT * FROM item_id order by item_id"
cursor1.execute(sql)
for row in cursor1.fetchall():
    if row[0] is not None:
       item_review.append([])
       continue
    item_review.append([])

cursor1.close()
print("Load item_id finished!")


#add item review
cursor2=connect.cursor()
sql ="SELECT * FROM user_review_more_than_30 order by item_id"
cursor2.execute(sql)
for row in cursor2.fetchall():
    # row[0]: usre_id,  row[2]:review, row[3]:item_id  
    str_cleaned=''
    if row[2] is not None:
        str_cleaned+=format_str(row[2],0)
    if str_cleaned=='':
        continue
    words= nlp.word_tokenize(str_cleaned)
    if row[3]<=len(item_review):
        item_review[row[3]-1]=item_review[row[3]-1]+words
cursor2.close()
print("Load item Reviews: Finished!")
item_size=len(item_review)
print("Item_size:%04d"%(item_size))

# test

# remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for text in item_review:
     for token in text:
         frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1]
          for text in item_review]

# train the model
item_review = [TaggedDocument(doc, [i]) for i, doc in enumerate(item_review)]
#print(documents)
model = Doc2Vec(item_review, vector_size=35, window=2, min_count=2, workers=5)
model.train(item_review,total_examples=model.corpus_count, epochs=150)
test_text=['Kevin','Smith','Strikes','Out']
inferred_vectors=model.infer_vector(test_text)
print(inferred_vectors)
sims=model.docvecs.most_similar([inferred_vectors],topn=10)
print(sims)
for count, sim in sims:
    sentence = item_review[count]
    words = ''
    for word in sentence[0]:
        words = words + word + ' '
    print(words, sim, len(sentence[0]))
model.save("item_doc2vec_item_size%04d.model"%(item_size))
#model.save("Doc2vec_Users.model")
print(model.docvecs.vectors_docs)
