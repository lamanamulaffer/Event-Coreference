import data_utils as data_utils
import tensorflow as tf
import numpy as np
import os
import sys
import time
import datetime
import pickle

def mark_sentence(s):
    x = s['sentence'].copy()
    x[s['start_index']] = "["+x[s['start_index']]
    x[s['end_index']] = x[s['end_index']] +"]"
    
    return " ".join(x)

with open("devset_200_ecb_coref_train.txt",'r', encoding="utf8") as f:
    sentences = f.readlines()
        

sentences = data_utils.read_file(sentences)

textloader = data_utils.TextLoader(sentences, 2, "data/enwiki_v2.model",demo=True)
x = textloader.features
y = textloader.labels
raw_data = textloader.raw_data

sess = tf.Session()

checkpoint_file = tf.train.latest_checkpoint('checkpoints/')
saver = tf.train.import_meta_graph(checkpoint_file + '.meta')
saver.restore(sess, checkpoint_file)

graph = tf.get_default_graph()
input_x = graph.get_operation_by_name("input_x").outputs[0]
predictions = graph.get_operation_by_name("accuracy/predictions").outputs[0]

correct = [0]
incorrect = [0]
unknown = [0]
i = 0
for x,y,r in zip(x,y,raw_data):
    predicted_coreference = \
    sess.run(predictions, feed_dict={input_x: np.array(x).reshape(1,len(x))})
    if predicted_coreference == y:
        correct.append("------Test " + str(i))
        s1 = mark_sentence(r[0])
        correct.append(s1.strip())
        
        s2 = mark_sentence(r[1])
        correct.append(s2.strip())

        correct.append("Predicted " + str(predicted_coreference) +  " Ground Truth: " + str(y) )
        correct[0] = correct[0] + 1
    elif predicted_coreference == 2:      
        unknown.append("------Test " + str(i))
        s1 = mark_sentence(r[0])
        unknown.append(s1.strip())
        
        s2 = mark_sentence(r[1])
        unknown.append(s2.strip())
        
        unknown.append("Predicted " + str(predicted_coreference) +  " Ground Truth: " + str(y))
        unknown[0] = unknown[0] + 1
    else:
        incorrect.append("------Test " + str(i))
        s1 = mark_sentence(r[0])
        incorrect.append(s1.strip())
        
        s2 = mark_sentence(r[1])
        incorrect.append(s2.strip())
        
        incorrect.append("Predicted " + str(predicted_coreference) +  " Ground Truth: " + str(y))
        incorrect[0] = incorrect[0] + 1
        
    i = i+1

print("unknown cases, total = ", unknown[0], "," , unknown[0]/i * 100 )
for line in unknown[1:]:
    #if line.startswith("----"):
    #    input()
    print(line)
    
print("incorrect cases, total = ", incorrect[0], "," , incorrect[0]/i * 100 )
for line in incorrect[1:]:
    #if line.startswith("----"):
    #    input()
    print(line)
    
print("correct cases, total = ", correct[0], "," , correct[0]/i * 100 )
for line in correct[1:]:
    print(line)







            
