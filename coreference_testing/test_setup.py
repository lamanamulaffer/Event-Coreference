import xmltodict
from textblob import Word
from nltk.corpus import stopwords
from pywsd.lesk import simple_lesk
import string
from copy import deepcopy

#Initializes all important content
#Loads test data file, caevo events for each sentences, all the nsenses & nevents & punctuations
def initialize():

    #1. Load input test data 
    # #all 
    test_file = 'checkpoints/data/combined_test.txt' 
    with open(test_file,'r', encoding="utf8") as f:
        sentences = f.readlines()
    test_file2 = 'checkpoints/data/ecb_incoref_cross_sentence.txt' 
    with open(test_file,'r', encoding="utf8") as f:
        sentences2 = f.readlines()
    #only coref
    test_file3 = 'checkpoints/data/ecb_coref_test.txt' 
    with open(test_file,'r', encoding="utf8") as f:
        sentences3 = f.readlines()

    #1.5 Arun's SRL file
    with open('checkpoints/data/srl_output.txt','r',encoding='utf8') as f:
        srl2 = f.readlines()
    srl2_lines = [] #3d list. list of chunks. each chunk is a list of rows
    srl2_chunk = [] #2d list. list of rows
    for line in srl2:
        row = line.split("\t")
        if (len(row) > 2):
            srl2_chunk.append(row)
        else:
            srl2_lines.append(srl2_chunk)
            srl2_chunk = []

    #2. Load caevo data
    caevo_sentences =[]
    for s in range(len(sentences)):
        cv_line = []
        with open('../caevo/data/data3/f_'+str(s)+'.txt.info.xml','r',encoding="utf8") as f:
            cur_caevo = xmltodict.parse(f.read())#open caevo xml file
            try:
                cv_events = cur_caevo['root']['file']['entry']['events']
                if (cv_events != None):
                    cv = cv_events['event']
                    if (cv != None):
                        for ev in cv:
                            try:
                                cv_line.append(ev['@string'].encode())
                            except:
                                cv_line.append(cv['@string'].encode())
            except:
                for cs in cur_caevo['root']['file']['entry']:
                    cv_events = cs['events']
                    if (cv_events != None):
                        cv = cv_events['event']
                        if (cv != None):
                            for ev in cv:
                                try:
                                    cv_line.append(ev['@string'].encode())
                                except:
                                    cv_line.append(cv['@string'].encode())

        #extend cv line with lth details also
        with open('../caevo/data/data3/f_'+str(s)+'.output','r') as f:
            tokens = f.readlines()
            tokens = list(map(lambda r: r.split("\t"),tokens))
            for tok_row in tokens:
                try:
                    if (tok_row[10] != '_'):
                        cv_line.append(tok_row[1].encode())
                except:
                    continue

        #extend cv_line with neural net details
        chunk = srl2_lines[s-1] #chunk corresp to cur sent
        for row in chunk:
            try:
                if (row[13] != '_'):
                    cv_line.append(row[1].encode())
            except:
                continue
        caevo_sentences.append(cv_line)

    #3. Load Noun senses lists; words, lemmatize v, lemmatize n, stem
    with open("eventive-noun-senses-ontonotes.txt",'r',encoding="utf8") as f:
        lines = f.readlines()[1:]

    nsenses, nsenses_stem, nsenses_lem_n, nsenses_lem_v = [],[], [], []
    for l in lines:
        li = l.split(" ")
        nsenses.append([li[0],int(li[1])])
        nsenses_stem.append([Word(li[0]).stem(), int(li[1])])
        nsenses_lem_n.append([Word(li[0]).lemmatize('n'),int(li[1])])
        nsenses_lem_v.append([Word(li[0]).lemmatize('v'),int(li[1])])
    nsenses_all = {'nsenses': nsenses, 'nsenses_stem': nsenses_stem,\
                    'nsenses_lem_n': nsenses_lem_n, 'nsenses_lem_v': nsenses_lem_v}

    #4. Load wordnet nouns with eventive noun senses
    with open("wordnet_event_nouns.txt",'r',encoding="utf8") as f:
        lines = f.readlines()[8:]
    nevents = []
    for l in lines:
        nevents.append(l.replace("\n",""))
    nevents_stem = list(map(lambda w: Word(w).stem(), nevents))
    nevents_lem_v = list(map(lambda w: Word(w).lemmatize('v'), nevents))
    nevents_lem_n = list(map(lambda w: Word(w).lemmatize('n'), nevents))
    nevents_all = {'nevents': nevents, 'nevents_stem': nevents_stem,\
                    'nevents_lem_v': nevents_lem_v, 'nevents_lem_n': nevents_lem_n}

    #5. Load string punctuation list
    punctuations = string.punctuation

    return sentences3, sentences2, sentences, caevo_sentences, nsenses_all, nevents_all, punctuations
                            
#For a given link(mention), generate all possible events. 
def get_event_guesses(link, nsenses, nevents, punctuation):
    tokens = link['sentence']
    cv_events = list(link['caevo_event'])
    guess, guess2, guess3 = [],[], [] #backups to avoid empty return
    for i in range(len(tokens)):
        guess3.append(i) 
        word = tokens[i].rstrip('\n').lower().replace("'","").replace("\"","")
        # print('main word:> ', word)
        wordS = Word(word).stem()
        wordLN = Word(word).lemmatize('n')
        wordLV = Word(word).lemmatize('v')
        if (word in punctuation or word in stopwords.words('english')):
            continue
        guess2.append(i) #all words except punctuations & stop words

        #1. check with caevo events
        for cv in cv_events:
            if (word == cv.decode('utf8').rstrip('\n').lower().replace("'","").replace("\"","")):
                guess.append(i)
                continue

        #2. check with noun events
        n, nstem, nlemn, nlemv = nevents['nevents'], nevents['nevents_stem'], nevents['nevents_lem_n'], nevents['nevents_lem_v']
        if (word in n or word in nstem or word in nlemn or word in nlemv or \
            wordS in n or wordS in nstem or wordS in nlemn or wordS in nlemv or \
            wordLN in n or wordLN in nstem or wordLN in nlemn or wordLN in nlemv or \
            wordLV in n or wordLV in nstem or wordLV in nlemn or wordLV in nlemv):
            guess.append(i)
            continue

        #3. check with noun senses
        n, nstem, nlemn, nlemv = nsenses['nsenses'], nsenses['nsenses_stem'], nsenses['nsenses_lem_n'], nsenses['nsenses_lem_v']
        lesk_syn = simple_lesk((" ").join(tokens), word)
        if (lesk_syn):
            lesk_list = lesk_syn.name().split('.')
            if (len(lesk_list) == 3):
                lsen = int(lesk_list[2])
                if ((word, lsen) in n or (wordS, lsen) in n or (wordLN, lsen) in n or (wordLV, lsen) in n or \
                    (wordS,lsen) in nstem or (wordLN,lsen) in nlemn or (wordLV,lsen) in nlemv):
                    guess.append(i)
                    continue
    return guess 

#for a chain (pair of mentions), return all the possible BF pairs
#chain has to be a pair of mentions
def genBFPairs(pair, nsenses, nevents, punctuation):
    s1, s2 = pair[0], pair[1]
    # print('orginal indices:> ', s1['start_index'], s2['start_index'])
    guess1 = get_event_guesses(s1, nsenses, nevents, punctuation)
    guess2 = get_event_guesses(s2, nsenses, nevents, punctuation)
    if (len(guess1) == 0):
        print('No guesses for m1:> ', pair[0]['raw_data'].encode())
        print()
    if (len(guess2) == 0):
        print('No guesses for m2:> ', pair[1]['raw_data'].encode())
        print()
    newchain = []
    for g1 in list(set(guess1)):
        for g2 in list(set(guess2)):
            s1_temp = deepcopy(s1)
            s2_temp = deepcopy(s2)
            s1_temp['start_index'], s1_temp['end_index'] = g1, g1
            s2_temp['start_index'], s2_temp['end_index'] = g2, g2
            #if the 2 mentions r from the same sent and they refer to the same word - ignore
            if (s1_temp['doc_id'] == s2_temp['doc_id'] and \
                s1_temp['sentence_index'] == s2_temp['sentence_index'] and \
                s1_temp['start_index'] == s2_temp['start_index']):
                continue
            else:
                newpair = [s1_temp, s2_temp]
                newchain.append(newpair)
    return newchain

#from test file - read lines and arrange into chains
def read_test_lines(lines, caevo_sentences):
    chains, this_chain, prev_chain = [], [], None
    for idx in range(len(lines)):
        line = lines[idx]
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
            cv_events = caevo_sentences[idx] #get the cv events for the sentence
            mention = {"raw_data":line,"pos_tag":pos_tag, \
                        "sentence_index": sentence_index, \
                        "start_index":start_index, \
                        "end_index":end_index, "sentence":sentence, \
                        "is_coref":is_coref,"doc_id":doc_id,\
                        "caevo_event": set(cv_events),\
                        "chain_index":chain_index,\
                        "link_index":link_index}
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
            print("line:> ",line.encode())

    chains.append(this_chain)
    return chains

#new chains are sorted into pairs: only include coref chains
def gen_new_coref_chains(chains):
    newchains = []
    for chain in chains:
        for pi in range(len(chain)):
            for pj in range(pi+1,len(chain)):
                # if (chain[pi]['is_coref'] == 0 or chain[pj]['is_coref'] == 0):
                #     continue #incoref - ignore
                if ((chain[pj]['sentence_index'] > chain[pi]['sentence_index']) or \
                   ((chain[pj]['sentence_index'] == chain[pi]['sentence_index']) and (chain[pj]['start_index'] > chain[pi]['start_index']))):
                    j = pj
                    i = pi
                else:
                    j = pi
                    i = pj
                newpair = [chain[i],chain[j]]
                newchains.append(newpair)
    return newchains

#original read files function
def read_file(lines):

    chains, this_chain, prev_chain = [], [], None
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
            "caevo_event":-1, \
            "ontonotes_event":-1,\
            "wordnet_event":-1}

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
            print(line.encode())

    chains.append(this_chain)          
    return chains

