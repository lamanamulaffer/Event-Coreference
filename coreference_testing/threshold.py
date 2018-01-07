
import test_setup_lamana as test_setup
import numpy as np

#return true if pair that is supposed to be positive is in the negative list or vice versa
def isNeg(m1, m2, pairlist, iscoref):
    d1, d2 = m1[1], m2[1]
    s1, s2 = m1[4], m2[4]
    c1, c2 = m1[2], m2[2]
    si1, si2 = m1[-3], m2[-3]
    for cp in range(len(pairlist)):
        mi = pairlist[cp][0]
        mj = pairlist[cp][0]
        if ((mi['is_coref'] == iscoref) and (mj['is_coref'] == iscoref)): #belongs to current data set
            continue
        if ((d1 == mi['doc_id']) and (s1 == mi['sentence_index']) and (c1 == mi['chain_index']) and (si1 == mi['start_index']) and\
            (d2 == mj['doc_id']) and (s2 == mj['sentence_index']) and (c2 == mj['chain_index']) and (si2 == mj['start_index'])):
            return True
    return False

modelN = int(input("Enter Model: Intra (0), Cross (1) or Mixed (2) :> "))
# threshold = float(input('Enter float threshold value :> '))
corefN = int(input("Analyze coref(0) or incoref data (1):> "))

if modelN == 0:
    modelT = "intra_model"
elif modelN == 1:
    modelT = "cross_model"
else:
    modelT = "mixed_model"

#for each threshold - calculate the different accuracies
sentences_coref, sentences_incoref, sentences, caevo_sentences, nsenses_all, nevents_all, punctuations = test_setup.initialize()
sentences = test_setup.read_test_lines(sentences, caevo_sentences) #organize into chains
sentences_incoref = test_setup.read_file(sentences_incoref)
sentences_coref = test_setup.read_file(sentences_coref)
pairlist = test_setup.gen_new_coref_chains(sentences) #organize into pairs
pairlist_incoref = test_setup.gen_new_coref_chains(sentences_incoref)
pairlist_coref = test_setup.gen_new_coref_chains(sentences_coref)


numCorefPairs = len(pairlist)
markedisNeg, pairs_considered, correct_marked, correct, incorrect, unknown, coref_marked = 0, 0, 0, 0, 0, 0, 0

print('len of all pairs, coref pairs, incoref pairs = ', len(pairlist), len(pairlist_coref), len(pairlist_incoref))
thresholdlist = [0.2, 0.475, 0.49, 0.5, 0.525, 0.6, 0.75, 0.9]
if corefN == 0:
    for threshold in thresholdlist:
        markedisNeg, pairs_considered, correct_marked, correct, incorrect, unknown, coref_marked = 0, 0, 0, 0, 0, 0, 0
        numCorefPairs = len(pairlist)
        for cp in range(0,203):
            #actual mentions
            act_m1 = pairlist[cp][0]
            act_m2 = pairlist[cp][1]
            try:
                #generated guesses
                logits_file = "checkpoints/logits/" + modelT + "/logits_for_sentence_" + str(cp) + ".txt"
                with open(logits_file, 'r', encoding='utf8') as f:
                    logits_sentences = f.readlines()
                #pick all mention pairs that have coref probs above threshold
                max_l, max_m1, max_m2 = None, None, None
                for idx in range(1,len(logits_sentences), 3):
                    #get m1, m2 and logits vals
                    l = logits_sentences[idx][1:-2].split(",")
                    m1 = l[:7]
                    m1.append(" ".join(l[7:]))
                    l = logits_sentences[idx+1][1:-2].split(",")
                    m2 = l[:7]
                    m2.append(" ".join(l[7:]))
                    logits = logits_sentences[idx+2].replace("[","").replace("]","").replace("\n","").split(" ")
                    logits = list(filter(lambda v: v!="", logits))
                    logits = list(map(lambda v: float(v), logits))
                    logits[-1] = threshold 
                    #is there some spike that is from the actual gs
                    if (int(m1[-3]) == act_m1['start_index'] and int(m2[-3]) == act_m2['start_index']):
                        pred = np.argmax(np.array(logits))
                        if pred == 1:
                            coref_marked = coref_marked + 1

                    #find best pair 
                    if (max_l == None):
                        max_l = logits
                        max_m1 = m1
                        max_m2 = m2
                    else:
                        if (max_l[1] < logits[1]):
                            max_l = logits
                            max_m1 = m1
                            max_m2 = m2
                
                #once we have the best pair - find the pred
                pred = np.argmax(np.array(max_l))
                if (pred == 1):
                    #is the highest spike and actual gs mention pair
                    if (int(max_m1[-3]) == act_m1['start_index'] and int(max_m2[-3]) == act_m2['start_index']):
                        correct_marked = correct_marked + 1
                    #check if highest spike belongs to negative data set
                    elif (isNeg(max_m1, max_m2, pairlist, 1)):
                        markedisNeg = markedisNeg + 1
                    correct = correct + 1
                elif (pred == 0):
                    incorrect = incorrect + 1
                else:
                    unknown = unknown + 1
                pairs_considered = pairs_considered + 1
            except:
                print('cant consider this')

        print('threshold = ', threshold)
        print('-------------------------')
        print('-------------------------')
        # print("Total # of pairs = ", numCorefPairs)
        # print("Unknown Details = ", unknown, unknown/numCorefPairs*100)
        # print("Incorrect Details = ", incorrect, incorrect/numCorefPairs*100)
        # print("Correct Details = ", correct, correct/numCorefPairs*100)
        # print("Correct Marked Details = ", correct_marked, correct_marked/numCorefPairs*100)
        # print("Coref Marked = ", coref_marked, coref_marked/correct*100)
        # print("Marked as Coref, Actually Negative = ", markedisNeg)
        # print('-------------------------')
        numCorefPairs = 203
        print("Total # of coref pairs = ", numCorefPairs)
        print("Unknown Details = ", unknown, unknown/numCorefPairs*100)
        print("Incorrect Details = ", incorrect, incorrect/numCorefPairs*100)
        print("Correct Details = ", correct, correct/numCorefPairs*100)
        print("Max spike, gs match = ", correct_marked, correct_marked/numCorefPairs*100)
        print("Some spike, gs match = ", coref_marked, coref_marked/correct*100)
        print("Marked as Coref, Actually Negative = ", markedisNeg)
        print('-------------------------')
        print('-------------------------')
elif corefN == 1: #incoref data set
    for threshold in thresholdlist:
        markedisNeg, pairs_considered, correct_marked, correct, incorrect, unknown, coref_marked = 0, 0, 0, 0, 0, 0, 0
        numCorefPairs = len(pairlist)
        for cp in range(203,numCorefPairs):
            #actual mentions
            act_m1 = pairlist[cp][0]
            act_m2 = pairlist[cp][1]
            try:
                #generated guesses
                logits_file = "checkpoints/logits/" + modelT + "/logits_for_sentence_" + str(cp) + ".txt"
                with open(logits_file, 'r', encoding='utf8') as f:
                    logits_sentences = f.readlines()
                #pick all mention pairs that have coref probs above threshold
                max_l, max_m1, max_m2 = None, None, None
                for idx in range(1,len(logits_sentences), 3):
                    #get m1, m2 and logits vals
                    l = logits_sentences[idx][1:-2].split(",")
                    m1 = l[:7]
                    m1.append(" ".join(l[7:]))
                    l = logits_sentences[idx+1][1:-2].split(",")
                    m2 = l[:7]
                    m2.append(" ".join(l[7:]))
                    logits = logits_sentences[idx+2].replace("[","").replace("]","").replace("\n","").split(" ")
                    logits = list(filter(lambda v: v!="", logits))
                    logits = list(map(lambda v: float(v), logits))
                    logits[-1] = threshold 
                    #is there some spike that is from the actual gs
                    if (int(m1[-3]) == act_m1['start_index'] and int(m2[-3]) == act_m2['start_index']):
                        pred = np.argmax(np.array(logits))
                        if pred == 0: #correct answer here is 0
                            coref_marked = coref_marked + 1

                    #find best pair 
                    if (max_l == None):
                        max_l = logits
                        max_m1 = m1
                        max_m2 = m2
                    else:
                        if (max_l[0] < logits[0]): #look at only incoref probs
                            max_l = logits
                            max_m1 = m1
                            max_m2 = m2
                
                #once we have the best pair - find the pred
                pred = np.argmax(np.array(max_l))
                if (pred == 0):
                    #is the highest spike and actual gs mention pair
                    if (int(max_m1[-3]) == act_m1['start_index'] and int(max_m2[-3]) == act_m2['start_index']):
                        correct_marked = correct_marked + 1
                    #check if highest spike belongs to positive data set
                    elif (isNeg(max_m1, max_m2, pairlist, 0)):
                        markedisNeg = markedisNeg + 1 #in this case what's marked as negative is actually positive
                    correct = correct + 1
                elif (pred == 1):
                    incorrect = incorrect + 1
                else:
                    unknown = unknown + 1
                pairs_considered = pairs_considered + 1
            except:
                print('cant consider this')
        print('threshold = ', threshold)
        print('-------------------------')
        print('-------------------------')
        # print("Total # of pairs = ", numCorefPairs)
        # print("Unknown Details = ", unknown, unknown/numCorefPairs*100)
        # print("Incorrect Details = ", incorrect, incorrect/numCorefPairs*100)
        # print("Correct Details = ", correct, correct/numCorefPairs*100)
        # print("Correct Marked Details = ", correct_marked, correct_marked/numCorefPairs*100)
        # print("InCoref Marked = ", coref_marked, coref_marked/correct*100)
        # print("Marked as InCoref, Actually positive = ", markedisNeg)
        # print('-------------------------')
        numCorefPairs = numCorefPairs - 203
        print("Total # of incoref pairs = ", numCorefPairs)
        print("Unknown Details = ", unknown, unknown/numCorefPairs*100)
        print("Incorrect Details = ", incorrect, incorrect/numCorefPairs*100)
        print("Correct Details = ", correct, correct/numCorefPairs*100)
        print("Max spike, gs match = ", correct_marked, correct_marked/numCorefPairs*100)
        print("Some spike, gs match = ", coref_marked, coref_marked/correct*100)
        print("Marked as InCoref, Actually Positive = ", markedisNeg)
        print('-------------------------')
        print('-------------------------')
            




        