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
documents=[]

# For users
# User profiles
# For users (put user profiles and reviews together as a user document)
cursor1=connect.cursor()
sql ="SELECT * FROM user_id order by user_id"
cursor1.execute(sql)
for row in cursor1.fetchall():
    #row[2]:selfdescriptions
    str_cleaned=''
    if row[2] is not None:
        str_cleaned=str_cleaned+format_str(row[2],0)
    if str_cleaned=='':
       documents.append([])
       continue
    words= nlp.word_tokenize(str_cleaned)
    documents.append(words)

cursor1.close()
print("Load User self_description: Finished!")

# User text feature = user self_description + user reviews + item name + item category

cursor2=connect.cursor()
sql ="SELECT * FROM user_review_more_than_30 order by user_id"
cursor2.execute(sql)
for row in cursor2.fetchall():
    # row[0]: usre_id,  row[2]:review   
    str_cleaned=''
    if row[2] is not None:
        str_cleaned+=format_str(row[2],0)
    if row[4] is not None:
        str_cleaned+=format_str(row[4],0)
    if row[6] is not None:
        str_cleaned+=format_str(row[5],0)
    if str_cleaned=='':
        continue
    words= nlp.word_tokenize(str_cleaned)
    if row[0]<=len(documents):
        documents[row[0]-1]=documents[row[0]-1]+words
cursor2.close()
print("Load User Reviews: Finished!")
user_size=len(documents)
print("User_size:%04d"%(user_size))

print(documents[7150])# test

# remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for text in documents:
     for token in text:
         frequency[token] += 1

texts = [[token for token in text if frequency[token] > 1]
          for text in documents]

# train the model
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(documents)]
#print(documents)
model = Doc2Vec(documents, vector_size=35, window=2, min_count=2, workers=5)
model.train(documents,total_examples=model.corpus_count, epochs=150)
test_text=['Silicon', 'bValley', 'Girlb', 'Apple', 'iPad', 'GB', 'yes', 'good', 'enough', 'to', 'make', 'me', 'put', 'down', 'my', 'Kindle', 'Pampers', 'Cruisers', 'Diapers', 'For', 'A', 'Snug', 'No', 'Gap', 'Fit', 'Nikon', 'Coolpix', 'Digital', 'Camera', 'Not', 'Exactly', 'an', 'SLR', 'but', 'Gets', 'the', 'Job', 'Done', 'Pampers', 'Swaddlers', 'for', 'Newborns', 'Confusing', 'for', 'the', 'First', 'Time', 'Mom', 'TiVo', 'Humax', 'DVDRRW', 'Series', 'Digital', 'Recorder', 'How', 'Do', 'I', 'Love', 'Thee', 'MAC', 'Shadestick', 'for', 'Eyes', 'Easy', 'as', 'Pumpkin', 'Pie', 'Postrio', 'Restaurant', 'San', 'Francisco', 'Still', 'Glamorous', 'Sonic', 'Rio', 'Sport', 'S', 'MP', 'Player', 'Hours', 'of', 'skipfree', 'music', 'for', 'Runners', 'Peter', 'Jacksons', 'Return', 'of', 'the', 'King', 'Tears', 'and', 'Triumph', 'Propel', 'Fitness', 'Water', 'Atkins', 'Friendlier', 'replacement', 'for', 'Gatorade', 'MAC', 'Paints', 'More', 'Than', 'Eyeshadow', 'philosophy', 'Amazing', 'Grace', 'BathShower', 'Gel', 'How', 'Sweet', 'The', 'Scent', 'MAC', 'Lipglass', 'Tints', 'For', 'Ultra', 'Shiny', 'Lips', 'not', 'a', 'wimpy', 'gloss', 'Chanel', 'Glossimers', 'meet', 'your', 'match', 'Mary', 'Kay', 'Signature', 'Lip', 'Gloss', 'Pria', 'Bar', 'Low', 'In', 'Points', 'for', 'those', 'on', 'Weight', 'Watchers', 'Revlon', 'Skinlights', 'Loose', 'Powder', 'Summer', 'Glow', 'All', 'Year', 'Long', 'Origins', 'A', 'Perfect', 'World', 'Body', 'Cream', 'Foils', 'Dry', 'Skin', 'Estee', 'Lauder', 'Illusionist', 'Maximum', 'Curling', 'Mascara', 'Meet', 'My', 'Straight', 'Asian', 'Lashes', 'Revlon', 'Skinlights', 'Lancome', 'Photogenic', 'knockoff', 'Origins', 'Perfect', 'World', 'You', 'Really', 'Really', 'Want', 'It', 'Sebastian', 'Potion', 'No', 'Put', 'a', 'little', 'Magic', 'back', 'into', 'your', 'life', 'Clinique', 'Body', 'Sloughing', 'Cream', 'Summer', 'Legs', 'or', 'Winter', 'Skin', 'Youre', 'Still', 'the', 'One', 'Fitness', 'Magazine', 'Mind', 'Body', 'Spirit', 'Blueflycom', 'Chic', 'but', 'not', 'cheap', 'Health', 'Magazine', 'Should', 'be', 'called', 'Health', 'and', 'Wellbeing', 'for', 'Women', 'benefits', 'benetint', 'lip', 'balm', 'Meet', 'the', 'Ugly', 'Stepsister', 'Restaurant', 'LuLu', 'Folsom', 'San', 'Francisco', 'Olay', 'A', 'Gender', 'Neutral', 'Beauty', 'Review', 'Cliniques', 'Almost', 'Lipstick', 'An', 'Exercise', 'in', 'High', 'Maintenance', 'Pretty', 'Cliniques', 'All', 'About', 'Eyes', 'Huh', 'ATThome', 'vs', 'DSL', 'What', 'to', 'expect', 'from', 'this', 'cable', 'modem', 'provider', 'Rivals', 'Quart', 'Slow', 'Cooker', 'My', 'Hero', 'Kiehls', 'Lip', 'Balm', 'v', 'Philosophy', 'Kiss', 'Me', 'Lip', 'Balm', 'Smackdown', 'Cliniques', 'Moisture', 'Surge', 'Treatment', 'An', 'Oilfree', 'Gel', 'for', 'My', 'Combination', 'Skin', 'Estee', 'Lauder', 'Perfectly', 'Clean', 'Foaming', 'Lotion', 'Cleanser', 'Lancomes', 'Mysterious', 'Vitabolic', 'BHA', 'Banishes', 'Blemishes', 'Give', 'the', 'Gift', 'of', 'Scurvy', 'Prevention', 'Neutrogena', 'Healthy', 'Skin', 'in', 'Shades', 'of', 'Pale', 'Chalky', 'Gray', 'Marie', 'Callenders', 'Brunchless', 'in', 'Sunnyvale', 'Estee', 'Lauder', 'Splash', 'Away', 'Cleanser', 'Little', 'Green', 'Clean', 'Bret', 'Lotts', 'Jewel', 'Taking', 'on', 'the', 'Almighty', 'Banana', 'Boat', 'Best', 'Bang', 'for', 'your', 'Buck', 'Roys', 'Poipu', 'Bar', 'Grill', 'Kauai', 'Hawaii', 'Maui', 'Ruths', 'Chris', 'Steakhouse', 'Lahaina', 'The', 'Sleek', 'Palm', 'Vx', 'Power', 'in', 'My', 'Pursedoes', 'it', 'come', 'in', 'pink', 'Luna', 'Bar', 'Rice', 'Crispy', 'Treats', 'in', 'Disguise', 'Bachelorette', 'Night', 'at', 'Barcelona', 'SF', 'La', 'Fondue', 'Fork', 'Feeding', 'Frenzy', 'in', 'Saratoga', 'CA', 'The', 'Best', 'Kept', 'Secret', 'in', 'Sunnyvale', 'Saucony', 'Shoe', 'Enough', 'for', 'the', 'Running', 'Challenged', 'Treat', 'Yourself', 'to', 'the', 'Watercourse', 'Way', 'in', 'Palo', 'Alto', 'Sony', 'Vaio', 'the', 'Porcelain', 'PC']
inferred_vectors=model.infer_vector(test_text)
print(inferred_vectors)
sims=model.docvecs.most_similar([inferred_vectors],topn=10)
print(sims)
for count, sim in sims:
    sentence = documents[count]
    words = ''
    for word in sentence[0]:
        words = words + word + ' '
    print(words, sim, len(sentence[0]))
model.save("Doc2vec_Users_plus_Items_Epinions_category_USize%04d.model"%(user_size))
#model.save("Doc2vec_Users.model")
print(model.docvecs.vectors_docs)
