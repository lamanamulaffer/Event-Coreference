#TESTING BF PAIRS!!!
from __future__ import print_function
import data_utils_intra_bf as data_utils_intra
import data_utils_cross_bf as data_utils_cross
import data_utils_mixed_bf as data_utils_mixed
import os
import tensorflow as tf
import numpy as np
import sys
import time
import datetime
import pickle
import io
import random
import test_setup_lamana as test_setup
from copy import deepcopy

def mark_sentence(s_in):
    s = deepcopy(s_in)
    x = s['sentence']
    x[s['start_index']] = "["+x[s['start_index']]
    x[s['end_index']] = x[s['end_index']] +"]"
    return " ".join(x)

modelN = int(input("Enter Model: Intra (0), Cross (1) or Mixed (2) :> "))
gpuN = input('Enter GPU number (0,1,2,3) :> ')
os.environ["CUDA_VISIBLE_DEVICES"] = gpuN


#setup correct models
sess = tf.Session()
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

#load graph
saver = tf.train.import_meta_graph(folder + metafile, clear_devices=True)
saver.restore(sess, folder + metafile.replace(".meta",""))
graph = tf.get_default_graph()
input_x = graph.get_operation_by_name("input_x").outputs[0]
predictions = graph.get_operation_by_name("accuracy/predictions").outputs[0]
new_logits = graph.get_operation_by_name("accuracy/new_logits").outputs[0]

print("Processing Final Test Set on ", modelT)

#input data + event details
_,sentences, caevo_sentences, nsenses_all, nevents_all, punctuations = test_setup.initialize()
sentences = test_setup.read_test_lines(sentences, caevo_sentences) #test lines + caevo
pairlist = test_setup.gen_new_coref_chains(sentences)
numpairs = len(pairlist)
print('Total number of sentences = ', len(sentences))
print('total number of coref mention pairs = ', len(pairlist))

#process each pair. for each pair - get list of guesses. pick most coref.
i = 0
correct_unmarked, correct_marked, incorrect, unknown = [0], [0], [0], [0]
b_step = 25
overall_len = len(pairlist)

for b in range(203,overall_len, b_step): 
    if (b+b_step >= overall_len):
        b_end = overall_len
    else:
        b_end = b + b_step
    for p in range(b,b_end):
        logits_rows = []
        save_logits_file = "checkpoints/logits/"+ modelT + "/logits_for_sentence_" + str(p) + '.txt'
        logits_rows.append(['is_coref', 'doc_id', 'chain_id', 'link_id', 'sent_id', 'start_id', 'end_id', 'raw_data'])
        print()
        print()
        print('Current pair #', p, 'in batch #', b)
        pair = pairlist[p]
        newpairlist = test_setup.genBFPairs(pair, nsenses_all, nevents_all, punctuations)
        act_m1 = mark_sentence(pair[0])
        act_m2 = mark_sentence(pair[1])

        #generate x, y, z data
        if modelN == 1:
            textloader = data_utils_cross.TextLoader(newpairlist, 2, "word2vec_enwiki/enwiki_v2.model", demo=True)
        elif modelN == 3:
            textloader = data_utils_mixed.TextLoader(newpairlist, 2, "word2vec_enwiki/enwiki_v2.model", demo=True)
        else:
            textloader = data_utils_intra.TextLoader(newpairlist, 2, "word2vec_enwiki/enwiki_v2.model", demo=True)
        x = textloader.features
        y = textloader.labels
        raw_data = textloader.raw_data

        #find the most suited pair in the guesses pair list
        print('generated logits')
        max_coref, max_row, max_y, max_raw_data = None, None, None, None
        for x,y,r in zip(x,y,raw_data):
            pred_logits = sess.run(new_logits, feed_dict={input_x: np.array(x).reshape(1,len(x))})
            row1 = [r[0]['is_coref'], r[0]['doc_id'], r[0]['chain_index'], r[0]['link_index'], r[0]['sentence_index'], r[0]['start_index'], r[0]['end_index'], mark_sentence(r[0]).strip().encode()]
            row2 = [r[1]['is_coref'], r[1]['doc_id'], r[1]['chain_index'], r[1]['link_index'], r[1]['sentence_index'], r[1]['start_index'], r[1]['end_index'], mark_sentence(r[1]).strip().encode()]
            logits_rows.append(row1)
            logits_rows.append(row2)
            logits_rows.append(pred_logits)

            if (max_coref == None):
                max_coref = pred_logits[0][1] #coref probs
                max_row = pred_logits
                max_y, max_raw_data = y, r
            else:
                if (max_coref < pred_logits[0][1]):
                    max_coref = pred_logits[0][1]
                    max_row = pred_logits
                    max_y, max_raw_data = y, r
        #save all logits rows in file
        ofile = open(save_logits_file, "w")
        for row in logits_rows:
            print(row, file=ofile)
            # print(row)
        ofile.close()



#ONLY SAVING LOGITS VALUES IN FILES

#         if (len(x) == 0):
#             print('cannot make prediction from pair #', p)
#         else:
#             #picked the highest coref spike out of all the options
#             best_pred = sess.run(tf.argmax(max_row,axis=1))
#             print('prediction made = ', best_pred[0], ' actual = ', max_y)
#             if best_pred[0] == max_y:
#                 if (max_raw_data[0]['start_index'] == pair[0]['start_index'] and max_raw_data[1]['start_index'] == pair[1]['start_index']):
#                     correct = correct_marked
#                 else:
#                     correct = correct_unmarked
#                 correct.append("------Test " + str(i))
#                 correct.append("Actual marked sentences ....")
#                 correct.append(act_m1.strip())
#                 correct.append(act_m2.strip())
#                 correct.append("Guessed marked sentences ...")
#                 s1 = mark_sentence(max_raw_data[0])
#                 correct.append(s1.strip())
#                 s2 = mark_sentence(max_raw_data[1])
#                 correct.append(s2.strip())
                
#                 correct.append("Predicted " + str(best_pred) +  " Ground Truth: " + str(max_y) )
#                 correct[0] = correct[0] + 1
#             elif best_pred[0] == 2:      
#                 unknown.append("------Test " + str(i))
#                 unknown.append("Actual marked sentences ....")
#                 unknown.append(act_m1.strip())
#                 unknown.append(act_m2.strip())
#                 unknown.append("Guessed marked sentences ...")
#                 s1 = mark_sentence(max_raw_data[0])
#                 unknown.append(s1.strip())
                
#                 s2 = mark_sentence(max_raw_data[1])
#                 unknown.append(s2.strip())
                
#                 unknown.append("Predicted " + str(best_pred) +  " Ground Truth: " + str(max_y))
#                 unknown[0] = unknown[0] + 1
#             else:
#                 incorrect.append("------Test " + str(i))
#                 incorrect.append("Actual marked sentences ....")
#                 incorrect.append(act_m1.strip())
#                 incorrect.append(act_m2.strip())
#                 incorrect.append("Guessed marked sentences ...")
#                 s1 = mark_sentence(max_raw_data[0])
#                 incorrect.append(s1.strip())
                
#                 s2 = mark_sentence(max_raw_data[1])
#                 incorrect.append(s2.strip())
                
#                 incorrect.append("Predicted " + str(best_pred) +  " Ground Truth: " + str(max_y))
#                 incorrect[0] = incorrect[0] + 1
#             i = i + 1
        
#     print('Batch Analysis ... ')
#     print("unknown cases out of all, total = ", unknown[0], "," , float(unknown[0])/i * 100 )
#     print("incorrect cases out of all, total = ", incorrect[0], "," , float(incorrect[0])/i * 100)
#     print("correct UNMARKED cases out of all, total = ", correct_unmarked[0], "," , float(correct_unmarked[0])/i * 100)
#     print("correct MARKED cases out of all, total = ", correct_marked[0], ",", float(correct_marked[0])/i*100)
#     print ('Total pairs', numpairs)

# print('-----------------------------')
# print('Final Analysis ... ')
# print("unknown cases out of all, total = ", unknown[0], "," , float(unknown[0])/i * 100 )
# print("incorrect cases out of all, total = ", incorrect[0], "," , float(incorrect[0])/i * 100)
# print("correct UNMARKED cases out of all, total = ", correct_unmarked[0], "," , float(correct_unmarked[0])/i * 100)
# print("correct MARKED cases out of all, total = ", correct_marked[0], ",", float(correct_marked[0])/i*100)
# print ('Total pairs', numpairs)


# of_unknown =open(folder + 'test_output/unknown_2.txt','w')
# of_correct_marked =open(folder + 'test_output/correct_marked_2.txt','w')
# of_correct_unmarked =open(folder + 'test_output/correct_unmarked_2.txt','w')
# of_incorrect =open(folder + 'test_output/incorrect_2.txt','w')

# print("unknown cases out of all, total = ", unknown[0], "," , float(unknown[0])/i * 100 ,file=of_unknown)
# for line in unknown[1:]:
#     try:
#         print (line.encode(),file=of_unknown)
#     except:
#         print ('Unicode error')
    
    
# print("incorrect cases out of all, total = ", incorrect[0], "," , float(incorrect[0])/i * 100, file=of_incorrect )
# for line in incorrect[1:]:
#     try:
#         print (line.encode(),file=of_incorrect)
#     except:
#         print ('Unicode error')
    
# print("marked correct cases out of all, total = ", correct_marked[0], "," , float(correct_marked[0])/i * 100,file=of_correct_marked )
# for line in correct_marked[1:]:
#     try:
#         print (line.encode(),file=of_correct_marked)
#     except:
#         print ('Unicode error')

# print("unmarked correct cases out of all, total = ", correct_unmarked[0], "," , float(correct_unmarked[0])/i * 100,file=of_correct_unmarked )
# for line in correct_unmarked[1:]:
#     try:
#         print (line.encode(),file=of_correct_unmarked)
#     except:
#         print ('Unicode error')

# of_unknown.close()
# of_incorrect.close()
# of_correct_marked.close()
# of_correct_unmarked.close()




            
