from __future__ import print_function
import data_utils_arun as data_utils_intra
import data_utils_cross as data_utils_cross
import data_utils_mixed as data_utils_mixed
import os
import tensorflow as tf
import numpy as np
import sys
import time
import datetime
import pickle
import io
import random

def mark_sentence(s):
    x = s['sentence']
    x[s['start_index']] = "["+x[s['start_index']]
    x[s['end_index']] = x[s['end_index']] +"]"
    return " ".join(x)

#read dev test file that has two line per chain
def read_dev_file(lines):
    chains = []
    this_chain = []
    for i in range(len(lines)):
        line = lines[i]
        try:
            tokens = line.split(" ")
            is_coref = int(tokens[0])
            doc_id = int(tokens[1])
            chain_index = int(tokens[2]) 
            link_index = int(tokens[3]) 
            sentence_index = int(tokens[4])
            pos_tag = tokens[5]
            start_index = int(tokens[6]) 
            end_index = int(tokens[7])
            sentence = tokens[8:]
            mention = {"raw_data":line,"pos_tag":pos_tag, \
                       "sentence_index": sentence_index, \
                       "start_index":start_index, \
                       "end_index":end_index, "sentence":sentence, \
                       "is_coref":is_coref,"doc_id":doc_id,\
                       "link_index":link_index,"chain_index":chain_index}
            if (len(this_chain) < 2):
                this_chain.append(mention) #add to current pair
            if (len(this_chain) == 2):
                chains.append(this_chain)
                this_chain = []
        except:
            print("line_skipped")         
    return chains

#read normal test file
def read_file(lines):
    chains = []
    this_chain = []
    prev_chain = None
    for line in lines:
        try:
            tokens = line.split(" ")
            is_coref = int(tokens[0])
            doc_id = int(tokens[1])
            chain_index = int(tokens[2]) 
            link_index = int(tokens[3]) 
            sentence_index = int(tokens[4])
            pos_tag = tokens[5]
            start_index = int(tokens[6]) 
            end_index = int(tokens[7])
            sentence = tokens[8:]
            mention = {"raw_data":line,"pos_tag":pos_tag, \
                       "sentence_index": sentence_index, \
                       "start_index":start_index, \
                       "end_index":end_index, "sentence":sentence, \
                       "is_coref":is_coref,"doc_id":doc_id,\
                       "link_index":link_index,"chain_index":chain_index}
            if prev_chain == None:
                prev_chain = chain_index
            if prev_chain == chain_index:
                this_chain.append(mention)
            else:
                chains.append(this_chain)
                this_chain = [mention]
                prev_chain = chain_index
        except:
            print("line_skipped")
    chains.append(this_chain)          
    return chains 

inputN = int(input("Enter Set type: Test (0) or Dev (1) :> "))
modelN = int(input("Enter Model: Intra (0), Cross (1) or Mixed (2) :> "))
gpuN = input('Enter GPU number (0,1,2,3) :> ')
os.environ["CUDA_VISIBLE_DEVICES"] = gpuN

# inputN = 1 #0 - test set, 1 - dev set
# modelN = 2 #0 - intra, 1 - cross, 2 - mixed
setT, modelT = None, None
#main set up
sess = tf.Session()
if inputN == 1: #using dev sets
    setT = "dev_set"
    if modelN == 0: #intra
        modelT = "intra_model"
        metafile = "model-6800.meta"
        folder = 'checkpoints/best_fold_intra_model/'
        with io.open(folder + 'developmentset_6_400_intra_doc.txt','r', encoding="utf8",errors="replace") as f:
            sentences = read_dev_file(f.readlines())
    elif modelN == 1:  #cross
        modelT = "cross_model"
        metafile = "model-19400.meta"
        folder = 'checkpoints/best_fold_cross_model/'
        with io.open(folder + 'best_fold_cross_test_set.txt','r', encoding="utf8",errors="replace") as f:
            sentences = read_dev_file(f.readlines())
    elif modelN == 2:
        modelT = "mixed_model"
        metafile = "model-5000.meta"
        folder = 'checkpoints/best_fold_mixed_model/'
        with io.open(folder + 'best_mixed_model_dev.txt','r', encoding="utf8",errors="replace") as f:
            sentences = read_dev_file(f.readlines())
elif inputN == 0: 
    
    #final test set
    with io.open('checkpoints/data/combined_test.txt','r', encoding="utf8",errors="replace") as f:
        sentences = read_file(f.readlines())
    setT = "test_set"
    # #dev set of intra model
    # with io.open('checkpoints/best_fold_intra_model/developmentset_6_400_intra_doc.txt','r', encoding="utf8",errors="replace") as f:
    #         sentences = read_dev_file(f.readlines())
    # setT = "test_set_from_fold"
    if modelN == 0:
        modelT = "intra_model"
        metafile = "model-7600.meta" 
        folder = 'checkpoints/best_overall_intra_model/'
    elif modelN == 1:
        modelT = "cross_model"
        metafile = "model-19400.meta"
        folder = 'checkpoints/best_fold_cross_model/'
    elif modelN == 2:
        modelT = "mixed_model"
        metafile = "model-5000.meta"
        folder = 'checkpoints/best_fold_mixed_model/'



#generate tensors, and x,y, raw data
if modelN == 1:
    texloader = data_utils_cross.TextLoader(sentences, 2, "word2vec_enwiki/enwiki_v2.model", demo=True)
elif modelN == 3: 
    texloader = data_utils_mixed.TextLoader(sentences, 2, "word2vec_enwiki/enwiki_v2.model", demo=True)
else:
    texloader = data_utils_intra.TextLoader(sentences, 2, "word2vec_enwiki/enwiki_v2.model", demo=True)
tensors = [texloader.features, texloader.labels, texloader.raw_data]
x = tensors[0]
y = tensors[1]
raw_data = tensors[2]

#open to write test results file
outputfolder = "checkpoints/test_output/"
output_file =open(outputfolder + modelT + "_" + setT + '_output.txt','w')
print("Processing ", setT, modelT, file=output_file)

#load graph
saver = tf.train.import_meta_graph(folder + metafile, clear_devices=True)
saver.restore(sess, folder + metafile.replace(".meta",""))
graph = tf.get_default_graph()
input_x = graph.get_operation_by_name("input_x").outputs[0]
predictions = graph.get_operation_by_name("accuracy/predictions").outputs[0]
new_logits = graph.get_operation_by_name("accuracy/new_logits").outputs[0]

correct = [0]
incorrect = [0]
unknown = [0]
i = 0
print("Reading ", setT, modelT)
print("Total Number of test examples = ", len(x))
for x,y,r in zip(x,y,raw_data):
    print('Processing Test Example #', i)
    predicted_coreference = sess.run(predictions, feed_dict={input_x: np.array(x).reshape(1,len(x))})
    newl = sess.run(new_logits, feed_dict={input_x: np.array(x).reshape(1,len(x))})
    if predicted_coreference == y:
        correct.append("------Test " + str(i))
        s1 = mark_sentence(r[0])
        correct.append(s1.strip())
        
        s2 = mark_sentence(r[1])
        correct.append(s2.strip())

        correct.append("Predicted " + str(predicted_coreference) +  " Ground Truth: " + str(y) + " Logits Coreference: " + str(newl[0][1]))
        correct[0] = correct[0] + 1
    elif predicted_coreference == 2:      
        unknown.append("------Test " + str(i))
        s1 = mark_sentence(r[0])
        unknown.append(s1.strip())
        
        s2 = mark_sentence(r[1])
        unknown.append(s2.strip())
        
        unknown.append("Predicted " + str(predicted_coreference) +  " Ground Truth: " + str(y) + " Logits Coreference: " + str(newl[0][1]))
        unknown[0] = unknown[0] + 1
    else:
        incorrect.append("------Test " + str(i))
        s1 = mark_sentence(r[0])
        incorrect.append(s1.strip())
        
        s2 = mark_sentence(r[1])
        incorrect.append(s2.strip())
        
        incorrect.append("Predicted " + str(predicted_coreference) +  " Ground Truth: " + str(y) + " Logits Coreference: " + str(newl[0][1]))
        incorrect[0] = incorrect[0] + 1
        
    i = i+1


print("unknown cases, total = ", unknown[0], "," , float(unknown[0])/i * 100 ,file=output_file)
for line in unknown[1:]:
    try:
        print (line.encode(),file=output_file)
    except:
        print ('Unicode error')
    
    
print("incorrect cases, total = ", incorrect[0], "," , float(incorrect[0])/i * 100, file=output_file )
for line in incorrect[1:]:
    try:
        print (line.encode(),file=output_file)
    except:
        print ('Unicode error')
    
print("correct cases, total = ", correct[0], "," , float(correct[0])/i * 100,file=output_file )
for line in correct[1:]:
    try:
        print (line.encode(),file=output_file)
    except:
        print ('Unicode error')

print ('Total pairs = ', i,file=output_file)

output_file.close()




            
