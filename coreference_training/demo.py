import data_utils

import tensorflow as tf
import numpy as np
import os
import sys
import time
import datetime
import pickle

CACHE_DIR = 'cache'



def mark_sentence(s):
    x = s['sentence']
    x[s['start_index']] = "["+x[s['start_index']]
    x[s['end_index']] = x[s['end_index']] +"]"
    return " ".join(x)

'''
print("Enter two sentences and the start and end position of the possible mentions:")
s1 = input('Enter sentence 1: \n')
m1_s = input('Enter mention start position: \n')
m1_e = input('Enter mention end position: \n')
s1_index = input('Enter sentence index in text: \n')
print()
s2 = input('Enter sentence 2: \n')
m2_s = input('Enter mention start position: \n')
m2_e = input('Enter mention end position: \n')
s2_index = input('Enter sentence index in text: \n')
print()

sentences = [{"sentence_index": int(s1_index), "start_index":int(m1_s),
              "end_index":int(m1_e), "sentence":s1.split(" ")},
             {"sentence_index": int(s2_index), "start_index":int(m2_s),
              "end_index":int(m2_e), "sentence":s2.split(" ")}]
'''
sentences = [[{"pos_tag":"NA", \
                       "sentence_index": 0, \
                       "start_index":3, \
                       "end_index":3, \
                       "sentence":"The test was failing".split(), \
                       "caevo_event":0, \
                       "ontonotes_event":0,\
                       "wordnet_event":0},
              {"pos_tag":"NA", \
                       "sentence_index": 1, \
                       "start_index":2, \
                       "end_index":3, \
                       "sentence":"test did not work because it was slow".split(), \
                       "caevo_event":0, \
                       "ontonotes_event":0,\
                       "wordnet_event":0}]]

model_path = "data/enwiki_v2.model"
textloader = data_utils.TextLoader(sentences,2,model_path, demo=True)

print (mark_sentence(sentences[0][0]))
print (mark_sentence(sentences[0][1]))
sess = tf.Session()

checkpoint_file = tf.train.latest_checkpoint('checkpoints/')
saver = tf.train.import_meta_graph(checkpoint_file + '.meta')
saver.restore(sess, checkpoint_file)

graph = tf.get_default_graph()
input_x = graph.get_operation_by_name("input_x").outputs[0]
predictions = graph.get_operation_by_name("accuracy/predictions").outputs[0]

predicted_coreference = \
    sess.run(predictions, feed_dict={input_x: np.array(textloader.features[0]).reshape(1,len(textloader.features[0]))})
print("Predicted coreference class is ", predicted_coreference)


    

