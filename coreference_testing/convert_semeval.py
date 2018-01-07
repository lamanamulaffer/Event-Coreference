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

modelN = int(input("Enter Model: Intra (0), Cross (1) or Mixed (2) :> "))
gpuN = input('Enter GPU number (0,1,2,3) :> ')
os.environ["CUDA_VISIBLE_DEVICES"] = gpuN

#main set up
sess = tf.Session() 
setT = "test_set"
with io.open('checkpoints/data/combined_test.txt','r', encoding="utf8",errors="replace") as f:
    sentences = read_file(f.readlines())

#list of all (doc_id,sent_id) tuples with assigned sentence index
d_s_ids, s_id_list, s_id = [], [], 0
for chain in sentences:
    for m in chain:
        if (m['doc_id'],m['sentence_index']) not in d_s_ids:
            d_s_ids.append((m['doc_id'],m['sentence_index']))
            s_id_list.append(s_id)
            s_id = s_id + 1
d_s_dict = dict(zip(d_s_ids, s_id_list))

if modelN == 0:
    modelT = "intra_model"
    metafile = "model-7600.meta" 
    folder = 'checkpoints/best_overall_intra_model/'
elif modelN == 1:
    modelT = "cross_model"
    metafile = "model-19400.meta"
    folder = 'checkpoints/best_fold_cross_model/'
else:
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

#load graph
saver = tf.train.import_meta_graph(folder + metafile, clear_devices=True)
saver.restore(sess, folder + metafile.replace(".meta",""))
graph = tf.get_default_graph()
input_x = graph.get_operation_by_name("input_x").outputs[0]
predictions = graph.get_operation_by_name("accuracy/predictions").outputs[0]

overall_list, chain_id, sent_id = [], 0, 0
#populate sent dictionary: clusters of mentions for each sentence
sent_dict = {}
for s in range(s_id):
    sent_dict[s] = []
doc_id_list = []
for x,y,r in zip(x,y,raw_data):
    s_id0 = d_s_dict[(r[0]['doc_id'],r[0]['sentence_index'])]
    s_id1 = d_s_dict[(r[1]['doc_id'],r[1]['sentence_index'])]
    doc_id_list.append(r[0]['chain_index']) #only one is enuf as it's intra doc test set
    predicted_coreference = sess.run(predictions, feed_dict={input_x: np.array(x).reshape(1,len(x))})
    if y == 1 and predicted_coreference[0] == 1: #actual says it's coref. include chain id here
        sent_dict[s_id0].append([r[0], r[0]['chain_index'], r[0]['start_index'], r[0]['end_index'], chain_id, chain_id])
        sent_dict[s_id1].append([r[1], r[1]['chain_index'], r[1]['start_index'], r[1]['end_index'], chain_id, chain_id])
       
    elif y == 1:
        sent_dict[s_id0].append([r[0], r[0]['chain_index'], r[0]['start_index'], r[0]['end_index'], chain_id, -1])
        sent_dict[s_id1].append([r[1], r[1]['chain_index'], r[1]['start_index'], r[1]['end_index'], chain_id, -1])
       
    elif predicted_coreference[0] == 1:
        sent_dict[s_id0].append([r[0], r[0]['chain_index'], r[0]['start_index'], r[0]['end_index'], -1, chain_id])
        sent_dict[s_id1].append([r[1], r[1]['chain_index'], r[1]['start_index'], r[1]['end_index'], -1, chain_id])
        
    else:
        sent_dict[s_id0].append([r[0], r[0]['chain_index'], r[0]['start_index'], r[0]['end_index'], -1, -1])
        sent_dict[s_id1].append([r[1], r[1]['chain_index'], r[1]['start_index'], r[1]['end_index'], -1, -1])
       
    chain_id = chain_id + 1 #each pair gets new id

doc_id_list = list(set(doc_id_list)) #unique set of doc ids
for dk in doc_id_list:
    doc_lines, s_id_final = [], 0
    doc_lines.append("#begin document (LuoTestCase);")
    for k,v in sent_dict.items():
        if len(v) == 0:
            continue
        if v[0][1] != dk: 
            continue #ignore sentence cluster that is not in current doc
        tokens = v[0][0]['sentence']
        for t in range(len(tokens)):
            row_tok = ['test'+str(s_id_final), '0', str(t), tokens[t].replace("\n","")]
            coref_tok = []
            for [m,d,si,ei,gsc,rsc] in v:
                thresh = gsc
                if t == si and t == ei and thresh != -1:
                    coref_tok.append("("+str(thresh)+")")
                elif t == si and thresh != -1:
                    coref_tok.append("("+str(thresh))
                elif t == ei and thresh != -1:
                    coref_tok.insert(0,str(thresh)+")")
            if len(coref_tok) == 0:
                coref_mention = "-"
            else:
                coref_mention = "|".join(coref_tok)
            row_tok.append(coref_mention)
            doc_lines.append("\t".join(row_tok))
        doc_lines.append("\n")
        s_id_final = s_id_final + 1 #update 
    del(doc_lines[-1])
    doc_lines.append("#end document")
    if (len(doc_lines) < 3):
        continue
    ofile = open("semeval_output_" + modelT + "/c_" + str(dk) + "_gs.txt","w",encoding="utf8")
    for dl in doc_lines:
        print(dl, file=ofile)
    ofile.close()





            
