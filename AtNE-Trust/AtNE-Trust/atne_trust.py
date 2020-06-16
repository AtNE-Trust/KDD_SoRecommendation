# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 12:01:30 2019

@author: 45016577
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 11:59:32 2018

@author: 45016577
"""

import pickle
import pymysql
import numpy as np
import random
from   gensim.models.doc2vec import Doc2Vec
import smart_open
import tensorflow as tf
import argparse
import logging 
import time
import networkx as nx
from sklearn import metrics
import matplotlib.pyplot as plt

#database connection
connect = pymysql.Connect(
    host='localhost',
    port=3306,
    user='root',
    passwd='123456',
    db='epinions_experiment',
    charset='utf8'
)
def get_user_related_itemID():
    user_related_itemID = []
    cursor2 = connect.cursor()
    sql ="SELECT * FROM user_id order by user_id"
    cursor2.execute(sql)
    for row in cursor2.fetchall():
        if row[0] is not None:
            user_related_itemID.append([])
            continue
        cursor2.close()
    cursor3=connect.cursor()
    sql ="SELECT * FROM user_review_more_than_30"
    cursor3.execute(sql)
    for row in cursor3.fetchall():
        # row[0]: usre_id,  row[3]:item_id  
        item_id  = row[3]
        if row[0]<=len(user_related_itemID):       
            user_related_itemID[row[0]-1]=user_related_itemID[row[0]-1]+[item_id]
    cursor3.close()
    user_related_itemID = np.array(user_related_itemID)
    return user_related_itemID

# user MF loading
fr = open('user_PMF_train_data.txt','rb') # load user latent feature matrix 由PMF得到的
user_PMF_data = pickle.load(fr)
fr.close()

# user_related_item_feature
fr1 = open('item_PMF_train_data.txt','rb')  # load item latent feature matrix 由PMF得到的
item_PMF_data = pickle.load(fr1)
fr1.close()

# item doc2vec
item_model = Doc2Vec.load("item_doc2vec_item_size21661.model")

item_list = get_user_related_itemID()
sorted_item_list = [] 
for item in item_list:
    new_item = list(set(item))
    new_item.sort(key=item.index)
    sorted_item_list.append(new_item)
sorted_item_list = np.array(sorted_item_list)  
i = 0
user_item_feature = [] 
for user_id in range(7151):
    user_related_item_feature = float(0)
    for item_id in sorted_item_list[user_id]: # trustor_related_item feature计算
        user_related_item_feature += item_PMF_data[item_id-1]
        user_related_item_feature += item_model.docvecs[item_id-1]
        i = i+2    
    user_related_item_feature = user_related_item_feature/i  
    user_item_feature.append(user_related_item_feature)  
    
#load doc2vec model to get review feature

model = Doc2Vec.load("Doc2vec_Users_plus_Items_Epinions_category_USize7151.model")

# load dataset and add negative sample

user_size = 7151
negNum = 1 #5
trustor_set = []
total_trust_pair=[] # [trustor,trustee] pair
train_trust_pair = []
trust_pair = []
#load train trust pair and negative instances
def load_trust_pair(args):
    global total_trust_pair
    global train_trust_pair
    start = time.time()
    cursor1=connect.cursor()
    sql ="SELECT * FROM user_trust_relation_time order by time limit 60"                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
    cursor1.execute(sql)
    for row in cursor1.fetchall():
        trustor = row[1]
        trustee = row[3]
        trustor_set.append(trustor)
        trust_pair.append([trustor,trustee])
        total_trust_pair.append([trustor,trustee,1])
        train_trust_pair.append([trustor,trustee,1])
    cursor1.close()
    G = nx.DiGraph()
    G.add_edges_from(trust_pair)
    for i in G.nodes():
        for j in G.nodes():
            if nx.has_path(G,i,j):
                #a = nx.shortest_path(G,i,j)
                b = nx.shortest_path_length(G,i,j)
                if b ==2 and G.degree(j) >80:
                    trustor_set.append(i)
                    train_trust_pair.append([i,j,1])
    
    global length
    length = len(train_trust_pair)
    end = time.time()
    
    for trustor in trustor_set:
        neglist=[]
        for t in range(negNum):
            j = np.random.randint(user_size+1)
            while [trustor,j] in total_trust_pair or j in neglist:
                j = np.random.randint(user_size+1)
            neglist.append(j)
            total_trust_pair.append([trustor,j,0])
            train_trust_pair.append([trustor,j,0])
    print("Creating negative instaces finished: %.2fs" % (time.time() - end))
    print("Loading total training trust pairs finished: %.2fs" % (end - start))
   
# load test pair and negative instances
test_trust_pair = []
test_trustor_set = []
cursor4 = connect.cursor()
sql = "SELECT pair_id,trustor_id,trustee_id FROM user_trust_relation_time WHERE pair_id not in (select pair_id from user_trust_relation_time_50) limit 30"
cursor4.execute(sql)
for row in cursor4.fetchall():
    trustor = row[1]
    trustee = row[2]
    test_trustor_set.append(trustor)
    test_trust_pair.append([trustor,trustee,1])
cursor4.close()
for test_trustor in test_trustor_set:
    test_neglist=[]
    for r in range(negNum):
        m = np.random.randint(user_size+1)
        while [test_trustor,m] in test_trust_pair or m in test_neglist:
            m = np.random.randint(user_size+1)
        test_neglist.append(m)
        test_trust_pair.append([test_trustor,m,0])

class Dataset:
    def __init__(self, args):
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.batch_size = args.batch_size
        self.total_trust_pair = total_trust_pair
        
        self.train_trust_pair = train_trust_pair
        self.test_trust_pair = test_trust_pair
        
        self.selected_pair = []
        self.total_trust_pair_size = len(self.total_trust_pair)
        self.flag = False
        
        self.test_pair = []
    def next_batch(self):
        # select a minibatch of trustor-trustee pairs 
        start = self.index_in_epoch  
        if self.epochs_completed == 0 and start == 0:
            random.shuffle(self.train_trust_pair)             
        if start >= length*2:
            self.epochs_completed +=1
            self.selected_pair = self.test_trust_pair
            self.index_in_epoch =0
            random.shuffle(self.selected_pair)
        else:
            self.index_in_epoch += self.batch_size 
            end = self.index_in_epoch
            self.selected_pair = self.train_trust_pair[start:end]
        trustor_PMF_in = [] 
        trustee_PMF_in = []
        trustor_PMF_item = []
        trustee_PMF_item = []
        trustor_review_in = []
        trustee_review_in = []
        
        trust_label_in = []
        
        positive_pair = []
    
        for trust_pair in self.selected_pair:
            trustor_id = trust_pair[0]
            trustee_id = trust_pair[1]
            trust_label = trust_pair[2]
            
            if trust_label == 1:
                positive_pair.append([trustor_id,trustee_id])
                
            
            trustor_feature = user_PMF_data[trustor_id - 1]
            trustee_feature = user_PMF_data[trustee_id - 1]
            
            trustoritem = user_item_feature[trustor_id - 1]
            trusteeitem = user_item_feature[trustee_id - 1]
            
            trustor_text_feature = model.docvecs[trustor_id - 1]
            trustee_text_feature = model.docvecs[trustee_id - 1]
            
            trustor_PMF_in.append(trustor_feature) 
            trustor_PMF_item.append(trustoritem)
            trustor_review_in.append(trustor_text_feature)
            
            trustee_PMF_in.append(trustee_feature) 
            trustee_PMF_item.append(trusteeitem)
            trustee_review_in.append(trustee_text_feature)
            
            trust_label_in.append(trust_label)
            
        nb_classes = 2
        targets = np.array(trust_label_in).reshape(-1)
        trust_label_in = np.eye(nb_classes)[targets]    
        
        return self.epochs_completed,self.index_in_epoch,trust_label_in,trustor_PMF_in,trustee_PMF_in,trustor_PMF_item,trustee_PMF_item,trustor_review_in,trustee_review_in,self.selected_pair,positive_pair
        
def main(args):
    # log
    logging.basicConfig(filename="log".format(time.strftime("%m-%d_%H_%M_%S", time.localtime())), 
    level=logging.INFO,format='%(asctime)s %(message)s\t',datefmt='%Y-%m-%d %H:%M:%S')    
    logging.info('begin to load data')
    print ('begin to train the model at ' + time.asctime())
    load_trust_pair(args) 
    dataset = Dataset(args)  
    
    print ('load data done at ' + time.asctime())
    logging.info('load data done')
    
    
    TRIGRAM_D = 35 
    n_hidden_1 = 30
    n_hidden_2 = 25
    n_hidden_3 = 20
    n_hidden_4 = 15
       
    # input placeholder
    trustor_rating_batch = tf.placeholder(tf.float32)  
    trustor_item_batch = tf.placeholder(tf.float32)
    trustor_review_batch = tf.placeholder(tf.float32)
    
    trustee_rating_batch = tf.placeholder(tf.float32) 
    trustee_item_batch = tf.placeholder(tf.float32)
    trustee_review_batch = tf.placeholder(tf.float32)
    
    trust_relation_known = tf.placeholder(tf.float32, shape=(None, 2))
    lr = 0.001 # learning rate
    
    
    
    # autoencoder-network
    weights_network = { 
        'encoder_h1': tf.Variable(tf.truncated_normal([TRIGRAM_D, n_hidden_1],)), 
        'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],)), 
        'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],)), 
        'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4],)), 
        'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_3],)), 
        'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2],)), 
        'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1],)), 
        'decoder_h4': tf.Variable(tf.truncated_normal([n_hidden_1, TRIGRAM_D],)), 
        } 
    biases_network = { 
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])), 
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])), 
        'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])), 
        'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])), 
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])), 
        'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])), 
        'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])), 
        'decoder_b4': tf.Variable(tf.random_normal([TRIGRAM_D])), 
        }
    
    #trustor encoder for rating
    layer_1_network_trustor = tf.nn.relu(tf.add(tf.matmul(trustor_rating_batch, weights_network['encoder_h1']), 
         biases_network['encoder_b1'])) 
    layer_2_network_trustor = tf.nn.relu(tf.add(tf.matmul(layer_1_network_trustor, weights_network['encoder_h2']), 
         biases_network['encoder_b2'])) 
    layer_3_network_trustor = tf.nn.relu(tf.add(tf.matmul(layer_2_network_trustor, weights_network['encoder_h3']), 
         biases_network['encoder_b3'])) 
    # 为了便于编码层的输出，编码层随后一层不使用激活函数 
    layer_4_network_trustor = tf.nn.relu(tf.add(tf.matmul(layer_3_network_trustor, weights_network['encoder_h4']), 
         biases_network['encoder_b4']))
    
    #trustee encoder for rating
    layer_1_network_trustee = tf.nn.relu(tf.add(tf.matmul(trustee_rating_batch, weights_network['encoder_h1']), 
         biases_network['encoder_b1'])) 
    layer_2_network_trustee = tf.nn.relu(tf.add(tf.matmul(layer_1_network_trustee, weights_network['encoder_h2']), 
         biases_network['encoder_b2'])) 
    layer_3_network_trustee = tf.nn.relu(tf.add(tf.matmul(layer_2_network_trustee, weights_network['encoder_h3']), 
         biases_network['encoder_b3'])) 
    # 为了便于编码层的输出，编码层随后一层不使用激活函数 
    layer_4_network_trustee =tf.nn.relu(tf.add(tf.matmul(layer_3_network_trustee, weights_network['encoder_h4']), 
         biases_network['encoder_b4']))
    
    #对于一对儿用户， encoder_rating的输出
    y_network_trustor = layer_4_network_trustor
    y_network_trustee = layer_4_network_trustee
  
    # autoencoder-rating
    weights_rating = { 
        'encoder_h1': tf.Variable(tf.truncated_normal([TRIGRAM_D, n_hidden_1],)), 
        'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],)), 
        'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],)), 
        'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4],)), 
        'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_3],)), 
        'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2],)), 
        'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1],)), 
        'decoder_h4': tf.Variable(tf.truncated_normal([n_hidden_1, TRIGRAM_D],)), 
        } 
    biases_rating = { 
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])), 
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])), 
        'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])), 
        'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])), 
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])), 
        'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])), 
        'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])), 
        'decoder_b4': tf.Variable(tf.random_normal([TRIGRAM_D])), 
        }
    
    #trustor encoder for rating
    layer_1_rating_trustor = tf.nn.relu(tf.add(tf.matmul(trustor_rating_batch, weights_rating['encoder_h1']), 
         biases_rating['encoder_b1'])) 
    layer_2_rating_trustor = tf.nn.relu(tf.add(tf.matmul(layer_1_rating_trustor, weights_rating['encoder_h2']), 
         biases_rating['encoder_b2'])) 
    layer_3_rating_trustor = tf.nn.relu(tf.add(tf.matmul(layer_2_rating_trustor, weights_rating['encoder_h3']), 
         biases_rating['encoder_b3'])) 
    # 为了便于编码层的输出，编码层随后一层不使用激活函数 
    layer_4_rating_trustor = tf.nn.relu(tf.add(tf.matmul(layer_3_rating_trustor, weights_rating['encoder_h4']), 
         biases_rating['encoder_b4']))
    
    #trustee encoder for rating
    layer_1_rating_trustee = tf.nn.relu(tf.add(tf.matmul(trustee_rating_batch, weights_rating['encoder_h1']), 
         biases_rating['encoder_b1'])) 
    layer_2_rating_trustee = tf.nn.relu(tf.add(tf.matmul(layer_1_rating_trustee, weights_rating['encoder_h2']), 
         biases_rating['encoder_b2'])) 
    layer_3_rating_trustee = tf.nn.relu(tf.add(tf.matmul(layer_2_rating_trustee, weights_rating['encoder_h3']), 
         biases_rating['encoder_b3'])) 
    # 为了便于编码层的输出，编码层随后一层不使用激活函数 
    layer_4_rating_trustee =tf.nn.relu(tf.add(tf.matmul(layer_3_rating_trustee, weights_rating['encoder_h4']), 
         biases_rating['encoder_b4']))
    
    #对于一对儿用户， encoder_rating的输出
    y_r_trustor = layer_4_rating_trustor
    y_r_trustee = layer_4_rating_trustee
  
    
    
    # autoencoder-review
    weights_review = { 
        'encoder_h1': tf.Variable(tf.truncated_normal([TRIGRAM_D, n_hidden_1],)), 
        'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],)), 
        'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],)), 
        'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4],)), 
        'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_3],)), 
        'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2],)), 
        'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1],)), 
        'decoder_h4': tf.Variable(tf.truncated_normal([n_hidden_1, TRIGRAM_D],)), 
        } 
    biases_review = { 
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])), 
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])), 
        'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])), 
        'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])), 
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])), 
        'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])), 
        'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])), 
        'decoder_b4': tf.Variable(tf.random_normal([TRIGRAM_D])), 
        }
    
    #trustor encoder for review
    layer_1_review_trustor = tf.nn.relu(tf.add(tf.matmul(trustor_review_batch, weights_review['encoder_h1']), 
         biases_review['encoder_b1'])) 
    layer_2_review_trustor = tf.nn.relu(tf.add(tf.matmul(layer_1_review_trustor, weights_review['encoder_h2']), 
         biases_rating['encoder_b2'])) 
    layer_3_review_trustor = tf.nn.relu(tf.add(tf.matmul(layer_2_review_trustor, weights_review['encoder_h3']), 
         biases_review['encoder_b3'])) 
    # 为了便于编码层的输出，编码层随后一层不使用激活函数 
    layer_4_review_trustor = tf.nn.relu(tf.add(tf.matmul(layer_3_review_trustor, weights_review['encoder_h4']), 
         biases_review['encoder_b4']))
    
    #trustee encoder for review
    layer_1_review_trustee = tf.nn.relu(tf.add(tf.matmul(trustee_review_batch, weights_review['encoder_h1']), 
         biases_rating['encoder_b1'])) 
    layer_2_review_trustee = tf.nn.relu(tf.add(tf.matmul(layer_1_review_trustee, weights_review['encoder_h2']), 
         biases_review['encoder_b2'])) 
    layer_3_review_trustee = tf.nn.relu(tf.add(tf.matmul(layer_2_review_trustee, weights_review['encoder_h3']), 
         biases_review['encoder_b3'])) 
    # 为了便于编码层的输出，编码层随后一层不使用激活函数 
    layer_4_review_trustee =tf.nn.relu(tf.add(tf.matmul(layer_3_review_trustee, weights_review['encoder_h4']), 
         biases_review['encoder_b4']))
    
    #对于一对儿用户， encoder_review的输出
    y_re_trustor = layer_4_review_trustor
    y_re_trustee = layer_4_review_trustee
    
    # autoencoder-item
    weights_item = { 
        'encoder_h1': tf.Variable(tf.truncated_normal([TRIGRAM_D, n_hidden_1],)), 
        'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],)), 
        'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],)), 
        'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4],)), 
        'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_3],)), 
        'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2],)), 
        'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1],)), 
        'decoder_h4': tf.Variable(tf.truncated_normal([n_hidden_1, TRIGRAM_D],)), 
        } 
    biases_item = { 
        'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])), 
        'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])), 
        'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])), 
        'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])), 
        'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])), 
        'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])), 
        'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])), 
        'decoder_b4': tf.Variable(tf.random_normal([TRIGRAM_D])), 
        }
    
    #trustor encoder for item
    layer_1_item_trustor = tf.nn.relu(tf.add(tf.matmul(trustor_item_batch, weights_item['encoder_h1']), 
         biases_item['encoder_b1'])) 
    layer_2_item_trustor = tf.nn.relu(tf.add(tf.matmul(layer_1_item_trustor, weights_item['encoder_h2']), 
         biases_item['encoder_b2'])) 
    layer_3_item_trustor = tf.nn.relu(tf.add(tf.matmul(layer_2_item_trustor, weights_item['encoder_h3']), 
         biases_item['encoder_b3'])) 
    # 为了便于编码层的输出，编码层随后一层不使用激活函数 
    layer_4_item_trustor = tf.nn.relu(tf.add(tf.matmul(layer_3_item_trustor, weights_item['encoder_h4']), 
         biases_item['encoder_b4']))
    
    #trustee encoder for review
    layer_1_item_trustee = tf.nn.relu(tf.add(tf.matmul(trustee_item_batch, weights_item['encoder_h1']), 
         biases_item['encoder_b1'])) 
    layer_2_item_trustee = tf.nn.relu(tf.add(tf.matmul(layer_1_item_trustee, weights_item['encoder_h2']), 
         biases_item['encoder_b2'])) 
    layer_3_item_trustee = tf.nn.relu(tf.add(tf.matmul(layer_2_item_trustee, weights_item['encoder_h3']), 
         biases_item['encoder_b3'])) 
    # 为了便于编码层的输出，编码层随后一层不使用激活函数 
    layer_4_item_trustee =tf.nn.relu(tf.add(tf.matmul(layer_3_item_trustee, weights_item['encoder_h4']), 
         biases_item['encoder_b4']))
    
    #对于一对儿用户， encoder_review的输出
    y_item_trsutor = layer_4_item_trustor
    y_item_trsutee = layer_4_item_trustee
    
    #data fusion unit
    z_p =  tf.Variable(tf.random_normal([n_hidden_4, n_hidden_4]))
    z_t = tf.Variable(tf.random_normal([n_hidden_4, n_hidden_4]))
    z_u = tf.Variable(tf.random_normal([n_hidden_4, n_hidden_4]))

    h_p = tf.Variable(tf.random_normal([n_hidden_4, n_hidden_4]))
    h_t = tf.Variable(tf.random_normal([n_hidden_4, n_hidden_4]))
    h_u = tf.Variable(tf.random_normal([n_hidden_4, n_hidden_4]))
    
    #trustor feature fusion 
    #p-rating t-review u-item
    h_r_re_trustor = tf.tanh(tf.matmul(y_r_trustor, h_p) + tf.matmul(y_re_trustor, h_t))
    h_r_item_trustor = tf.tanh(tf.matmul(y_r_trustor, h_p) + tf.matmul(y_item_trsutor, h_u))
    h_re_item_trustor = tf.tanh(tf.matmul(y_re_trustor, h_t) + tf.matmul(y_item_trsutor, h_u))                  
                                       
    z_r_re_trustor = tf.sigmoid(tf.matmul(y_r_trustor, z_p) + tf.matmul(y_re_trustor, z_t))
    z_r_item_trustor = tf.sigmoid(tf.matmul(y_r_trustor, z_p) + tf.matmul(y_item_trsutor, z_u))
    z_re_item_trustor = tf.sigmoid(tf.matmul(y_re_trustor, z_t) + tf.matmul(y_item_trsutor, z_u))

    r_trustor = (1 - z_re_item_trustor) * h_re_item_trustor + z_re_item_trustor * y_r_trustor
    re_trustor = (1 - z_r_item_trustor) * h_r_item_trustor + z_r_item_trustor * y_re_trustor
    item_trustor = (1 - z_r_re_trustor) * h_r_re_trustor + z_r_re_trustor * y_item_trsutor

    #trustee feature fusion 
    #p-rating t-review u-item
    h_r_re_trustee = tf.tanh(tf.matmul(y_r_trustee, h_p) + tf.matmul(y_re_trustee, h_t))
    h_r_item_trustee = tf.tanh(tf.matmul(y_r_trustee, h_p) + tf.matmul(y_item_trsutee, h_u))
    h_re_item_trustee = tf.tanh(tf.matmul(y_re_trustee, h_t) + tf.matmul(y_item_trsutee, h_u))                  
                                       
    z_r_re_trustee = tf.sigmoid(tf.matmul(y_r_trustee, z_p) + tf.matmul(y_re_trustee, z_t))
    z_r_item_trustee = tf.sigmoid(tf.matmul(y_r_trustee, z_p) + tf.matmul(y_item_trsutee, z_u))
    z_re_item_trustee = tf.sigmoid(tf.matmul(y_re_trustee, z_t) + tf.matmul(y_item_trsutee, z_u))

    r_trustee = (1 - z_re_item_trustee) * h_re_item_trustee + z_re_item_trustee * y_r_trustee
    re_trustee = (1 - z_r_item_trustee) * h_r_item_trustee + z_r_item_trustee * y_re_trustee
    item_trustee = (1 - z_r_re_trustee) * h_r_re_trustee + z_r_re_trustee * y_item_trsutee

    # decoder-rating 
    def decoder_rating(x): 
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights_rating['decoder_h1']), 
         biases_rating['decoder_b1'])) 
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights_rating['decoder_h2']), 
         biases_rating['decoder_b2'])) 
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights_rating['decoder_h3']), 
        biases_rating['decoder_b3'])) 
        layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights_rating['decoder_h4']), 
        biases_rating['decoder_b4'])) 
        return layer_4 
    decoder_rating_trustor = decoder_rating(r_trustor)
    decoder_rating_trustee = decoder_rating(r_trustee)
    
    # decoder-review 
    def decoder_review(x): 
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights_review['decoder_h1']), 
         biases_review['decoder_b1'])) 
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights_review['decoder_h2']), 
         biases_review['decoder_b2'])) 
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights_review['decoder_h3']), 
        biases_review['decoder_b3'])) 
        layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights_review['decoder_h4']), 
        biases_review['decoder_b4'])) 
        return layer_4 
    decoder_review_trustor = decoder_review(re_trustor)
    decoder_review_trustee = decoder_review(re_trustee)
    
    # decoder-review 
    def decoder_item(x): 
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights_item['decoder_h1']), 
         biases_item['decoder_b1'])) 
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights_item['decoder_h2']), 
         biases_item['decoder_b2'])) 
        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights_item['decoder_h3']), 
        biases_item['decoder_b3'])) 
        layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights_item['decoder_h4']), 
        biases_item['decoder_b4'])) 
        return layer_4 
    decoder_item_trustor = decoder_item(item_trustor)
    decoder_item_trustee = decoder_item(item_trustee)
    
    #loss for rating encoder
    y_pred_rating_trustor=decoder_rating_trustor
    y_true_rating_trustor=trustor_rating_batch
    
    y_pred_rating_trustee=decoder_rating_trustee
    y_true_rating_trustee=trustee_rating_batch
    
    cost_rating_trustor = tf.reduce_mean(tf.pow(y_true_rating_trustor - y_pred_rating_trustor, 2)) 
    cost_rating_trustee = tf.reduce_mean(tf.pow(y_true_rating_trustee - y_pred_rating_trustee, 2))
    cost_rating = cost_rating_trustor + cost_rating_trustee
    
    #loss for review encoder
    y_pred_review_trustor=decoder_review_trustor
    y_true_review_trustor=trustor_review_batch
    
    y_pred_review_trustee=decoder_review_trustee
    y_true_review_trustee=trustee_review_batch
    
    cost_review_trustor = tf.reduce_mean(tf.pow(y_true_review_trustor - y_pred_review_trustor, 2)) 
    cost_review_trustee = tf.reduce_mean(tf.pow(y_true_review_trustee - y_pred_review_trustee, 2))
    cost_review = cost_review_trustor + cost_review_trustee
    
    
    #loss for item encoder
    y_pred_item_trustor=decoder_item_trustor
    y_true_item_trustor=trustor_item_batch
    
    y_pred_item_trustee=decoder_item_trustee
    y_true_item_trustee=trustee_item_batch
    
    cost_item_trustor = tf.reduce_mean(tf.pow(y_true_item_trustor - y_pred_item_trustor, 2)) 
    cost_item_trustee = tf.reduce_mean(tf.pow(y_true_item_trustee - y_pred_item_trustee, 2))
    cost_item = cost_item_trustor + cost_item_trustee
    
    #loss for prediction
    trustor_out = tf.concat([r_trustor,re_trustor,item_trustor],1) 
    trustee_out = tf.concat([r_trustee,re_trustee,item_trustee],1)
    trust_feature_combine =  tf.concat([trustor_out,trustee_out],1)
    
    w_N = tf.Variable(tf.random_normal(shape=(90, 2), name='w_N'))
    b_N = tf.Variable(tf.random_normal(shape=(1,2)), name='b_N')
    prediction = tf.matmul(trust_feature_combine, w_N) + b_N
    
    cost_final = tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = trust_relation_known) #cross-entropy loss function
    loss = cost_rating + cost_review + cost_final + cost_item
    loss = tf.multiply(loss, 0.0001)
    
    correct_prediction_final = tf.equal(tf.argmin(prediction, 1), tf.argmin(trust_relation_known, 1))  
    
    accuracy_y_final = tf.reduce_mean(tf.cast(correct_prediction_final, "float")) * tf.constant(100.0)
    argmax_prediction = tf.argmax(prediction, 1)
    argmax_y = tf.argmax(trust_relation_known, 1)
    TP = tf.count_nonzero(argmax_prediction * argmax_y, dtype=tf.float32)
    TN = tf.count_nonzero((argmax_prediction - 1) * (argmax_y - 1), dtype=tf.float32)
    FP = tf.count_nonzero(argmax_prediction * (argmax_y - 1), dtype=tf.float32)
    FN = tf.count_nonzero((argmax_prediction - 1) * argmax_y, dtype=tf.float32)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    
    
    # train step optimizer
    train_step = tf.train.AdamOptimizer(lr).minimize(loss)  
    config = tf.ConfigProto() 
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        epoch = 0
        tr_loss = 0
        step = 0
        p = 0
        while epoch <= args.niter :
            epochs_completed,index_in_epoch,trust_label_batch,\
            trustor_PMF_batch,trustee_PMF_batch,\
            trustor_PMF_item_batch,trustee_PMF_item_batch,\
            trustor_review_in_batch,trustee_review_in_batch, testing_pair,positive_pair_test= dataset.next_batch() #testing_pair所有的test pair。positive_pair_test： label为 1 的pair
            #print(positive_pair_test)
            if epochs_completed == epoch + 1:               
                #training loss
                tr_loss /= args.trainingset_size
                #print("Epoch #%-2d Finished | Train Loss: %-4.3f" % (epoch, tr_loss)) 
                #logging.info('Epoch_%04d_finished\ttraing_loss = %.6f'%(epoch, tr_loss))
                tr_loss = 0               
                #test step
                test_loss,test_prediction,f = sess.run([loss,prediction,f1],feed_dict = {trustor_rating_batch:trustor_PMF_batch,trustor_item_batch:trustor_PMF_item_batch,
                                                 trustor_review_batch:trustor_review_in_batch,trustee_rating_batch:trustee_PMF_batch,
                                                 trustee_item_batch:trustee_PMF_item_batch,trustee_review_batch:trustee_review_in_batch,
                                                 trust_relation_known:trust_label_batch})
                test_size = len(test_trust_pair)
                test_loss/=test_size
                step = 0
                print(f)
                #test_auc = metrics.roc_auc_score(trust_relation_known,test_prediction)
                #print("Epoch #%-2d | F1 score: %f | Accuracy: %f" %(epoch, f1, accuracy_y_final))
                epoch += 1
            else:
                #train step
                p = p+1
                _, _tr_loss,cost_test = sess.run([train_step,loss,weights_rating['encoder_h2']],feed_dict = {trustor_rating_batch:trustor_PMF_batch,trustor_item_batch:trustor_PMF_item_batch,
                                                 trustor_review_batch:trustor_review_in_batch,trustee_rating_batch:trustee_PMF_batch,
                                                 trustee_item_batch:trustee_PMF_item_batch,trustee_review_batch:trustee_review_in_batch,
                                                 trust_relation_known:trust_label_batch})
                                                
                tr_loss+=_tr_loss
                step = step + 1
            
                
                
                #print("Epoch #%-2d | Step: %-2d | Train Loss: %f" % (epoch, step, tr_loss)) 
    print ('Finish at ' + time.asctime())            
    logging.info('done')
                
if __name__ == '__main__':
    # parse argv
    parser = argparse.ArgumentParser(description = 'trust_prediction_model')
    parser.add_argument('--data_root', action='store', dest='data_root', default='data/')
    parser.add_argument('--save_dir', action='store', dest='save_dir', default='log/')

    parser.add_argument('--niter', action='store', dest='niter', default=5, type=int)
    parser.add_argument('--trainingset_size', action='store', dest='trainingset_size', default=500000, type=int)
    parser.add_argument('--batch_size', action='store', dest='batch_size', default=30, type=int)#batch size
    
    parser.add_argument('--user_size', action='store', dest='user_size', default=7151, type=int)
    
    parser.add_argument('--lr', action='store', dest='lr', default=0.0001, type=float)#0.001
    parser.add_argument('--decay', action='store', dest='decay', default=1.1, type=float)
    args=parser.parse_args()
    main(args)
