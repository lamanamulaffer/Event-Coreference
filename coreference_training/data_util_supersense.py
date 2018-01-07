# This file reads the data in the data/sst directory
# The improvement is insignificant (see log 17)
import os
import numpy as np
import pickle
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.parsing.preprocessing import STOPWORDS
from textblob import Word
from textblob.wordnet import Synset
import spacy

#buckets for wordnet similarity and cosine similarity
#the buckets are created manually in a preprocessing step
# the last bucket is for unknown similarities 
cosine_bucket = [-1, 0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.9, 1.0, None]
wordnet_bucket = [0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.26,0.34,0.5,1, None]
supersense_list = []
with open("data\sst\WNSS_07.TAGSET","r") as f:
    supersense_list = f.read().split()
supersense_list.append("0")         
print(supersense_list)

class TextLoader():
    def __init__(self,sentences,window,model_path,tensor_path=None, demo=False):
            
        self.window = window
        self.nlp = spacy.load('en')
        #WNSS_07.TAGSET
    
        if tensor_path is not None and os.path.exists(tensor_path):
            print("Loading saved tensors...")
            self.load_tensors(tensor_path)
        else:
            self.load_model(model_path)
            print("Generating tensors...")
            if demo:
                self.features,self.labels,self.raw_data = self.gen_features_and_labels(sentences)                
            else:
                self.features,self.labels,self.raw_data = {},{},{}
                self.features['coref'],self.labels['coref'],self.raw_data['coref'] = self.gen_features_and_labels(sentences['coref'])
                self.features['incoref'],self.labels['incoref'],self.raw_data['incoref'] = self.gen_features_and_labels(sentences['incoref'])
            if tensor_path is not None:
                self.save_tensors(tensor_path)
                
        self.features_size = ( 2 * (2*self.window+1) * self.model_shape +  # size of the window (Contextual features)
                               #2 * 3 + #event information (Event Features)
                               1 +  # distance between sentences (Relational Features)
                               1 * len(cosine_bucket) + # hotvector of mention headword similarity and avg mention similarity (Relational Features)
                               2 * len(wordnet_bucket) +
                               2 * (int(len(supersense_list)/2) + 1)) # hotvector of mention headword similarity and hypernyms (Relational Features)
    
    #Loading the word2vec model
    def load_model(self, model_path):
        print("Loading word2vec model...")
        word2vec_model = word2vec.Word2Vec.load(model_path)
        #word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        self.model = word2vec_model
        self.model_shape = self.model['news'].shape[0]      

    def save_tensors(self, tensors_path):
        tensors = [self.features, self.labels,self.model_shape,self.raw_data]
        with open(tensors_path, 'wb') as f:
            pickle.dump(tensors, f)


    def load_tensors(self, tensors_path):
        with open(tensors_path, 'rb') as f:
            tensors = pickle.load(f)
        self.features = tensors[0]
        self.labels = tensors[1]
        self.model_shape = tensors[2]
        self.raw_data = tensors[3]

    # input: list of words and a window(k)
    # returns the first k words that occurs in the model
    # returns a row of size k*model_shape
    def k_words_embeddings(self,words_list,k):
        row = []
        i = 0
        for e in words_list:
            if i >= k:
                break
            elif e in self.model:
                row.extend(self.model[e.lower()].tolist())
                i = i + 1
        while i < k:
            row.extend([0]*self.model_shape) #out of bounds
            i = i + 1
        return row


    def wordlist_average(self,wordlist):
        total = np.zeros((self.model_shape,),dtype="float32")
        nwords = 0
        for word_i in wordlist:
            word = word_i.lower()
            if word in self.model:
                total = np.add(total, self.model[word])
                nwords = nwords + 1
        if nwords < 1:
            return None
        else:
            return np.divide(total,nwords).tolist()



    def get_word2vec_sim(self,a1,a2):
        if a1.lower() in self.model and a2.lower() in self.model:
            return self.model.similarity(a1.lower(),a2.lower())
        else:
            return None
            
    #Intended for similarity of the mentions
    def cosine_similarity(self,a,b):
        return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
            
    def get_max_synsets(self,synsets1,synsets2):
        sim_max = None
        max_syn1 = None
        max_syn2 = None
        for synset1 in synsets1:
            for synset2 in synsets2:
                sim = synset1.path_similarity(synset2) 
                if sim_max == None or (sim != None and sim > sim_max):
                    sim_max = sim
                    max_syn1 = synset1
                    max_syn2 = synset1
                    
        return sim_max,max_syn1,max_syn2
          
    def get_wordnet_sim(self,a1,a2):             
        w1 = Word(a1)
        w2 = Word(a2)
         
        sim_max , max_syn1 , max_syn2 = self.get_max_synsets(w1.synsets,w2.synsets)
        hyper_sim = None
        if sim_max != None:
            hyper_sim, z, z = self.get_max_synsets(max_syn1.hypernyms(),max_syn2.hypernyms())
       
        return sim_max,hyper_sim
    
    def get_bucket(self,t, x):
        counter = [0] *len(t)
        if x == None:
            counter[len(t)-1] = 1
            return counter
        i = 0
        while i < len(t) - 2 and t[i] < float(x):
            i = i + 1
        counter[i] = counter[i] + 1
        return counter

    def get_supersense(self,sent):
        mention = sent['ss_tag'][sent['start_index']:sent['end_index']+1]
        final = [0] * (int(len(supersense_list)/2) + 1)
        for tag in mention:
            idx = int(supersense_list.index(tag.strip())/2)
            final[idx] = final[idx] + 1
        return final
        
            
    # generates the embeddings of the the mention and the window arounds it
    # generates the head word
    # return the embeddings of the window.
    # If none of the words of the mention appear in the model, it returns none (can't be computed)
    def window_embeddings(self,sentence,s,e):
        row = []
        before_mention = sentence[:s]
        after_mention = sentence[e+1:]
        mention =sentence[s:e+1]
        
        #getting the headword
        doc = self.nlp(" ".join(mention))
        sents = list(doc.sents)
        head_word = sents[0].root
        
        row.extend(self.k_words_embeddings(reversed(before_mention),self.window))
        m_avg = self.wordlist_average(mention)
        
        if m_avg == None:
            return None,None,None
        else:
            #Adding the mention's average embeddings
            row.extend(m_avg)
            #Adding the headword embedding
            #row.extend(self.words_vec([head_word],1))
            row.extend(self.k_words_embeddings(after_mention,self.window))
            
            return row, str(head_word).lower(), m_avg

    # creates one row of input for the neural network
    # return None, if the none of the words in the mention are in the word2vec model
    def get_row(self,sentence_i, sentence_j):
        data_row = []            
        # appending the vecotrs for the chosen window for both sentences
        data_row_i, headword_i, avg_i = self.window_embeddings(sentence_i['sentence'],sentence_i['start_index'], sentence_i['end_index'])
        data_row_j ,headword_j, avg_j = self.window_embeddings(sentence_j['sentence'],sentence_j['start_index'],sentence_j['end_index'])
        if avg_i == None or avg_j == None:
            return None
        
        else:   
            # appending the vecotrs for the chosen window for sentences_i
            data_row.extend(data_row_i)
                          
            #repeat for j
            data_row.extend(data_row_j)
            
            
            # adds any additional features
            #       feature 1: distance between senteces 
            data_row.append(sentence_j['sentence_index'] - sentence_i['sentence_index'])
            
            #       feature 2: cosine sim of headwords of mentions
            cosine_sim = self.get_word2vec_sim(headword_i,headword_j)
            data_row.extend(self.get_bucket(cosine_bucket, cosine_sim))
            
            #       feature 3: cosine sim of average mentions
            #avg_cosine_sim = self.cosine_similarity(avg_i,avg_j)
            
            #data_row.extend(self.get_bucket(cosine_bucket, avg_cosine_sim))
            #       feature 4 & 5 : wordnet sim of mentions and their hypernyms
            wordnet_sim,wordnet_hypersim = self.get_wordnet_sim(headword_i,headword_j)
            data_row.extend(self.get_bucket(wordnet_bucket, wordnet_sim))
            data_row.extend(self.get_bucket(wordnet_bucket, wordnet_hypersim))

            #       feature 6: supersenses of the mentions (of each sentence)
            data_row.extend(self.get_supersense(sentence_i))
            data_row.extend(self.get_supersense(sentence_j))

            
            return data_row
        
    # The following is used to generate the 'raw_data'
    def get_raw_data(self,link,keys,values):
        mention = link.copy()
        for k,v in zip(keys,values):
            mention[k] = v
        mention['raw_data'] = self.get_mention_line(mention)
        return mention
        
    def get_mention_line(self,mention):
        data = ["is_coref","doc_id","chain_index","link_index", \
                "sentence_index","pos_tag","start_index","end_index"]
                #"caevo_event","ontonotes_event","wordnet_event"]
        mention_s = ""
        for key in data:
            mention_s = mention_s + str(mention[key]) + " "
        
        mention_s = mention_s + " ".join(mention["raw_data"].split(" ")[8:]).strip()
        return mention_s.rstrip()
    
    def gen_features_and_labels(self,chains,data=""):
        x = []
        y = []
        raw_data = []
        row_id = 0
        for chain in chains:
            for pi in range(len(chain)):
                for pj in range(pi+1,len(chain)):
                    if row_id % 500 == 0:
                        print("Processing row ", row_id,"...")
                    #i should preceed j
                    if chain[pj]['sentence_index'] > chain[pi]['sentence_index'] or \
                       (chain[pj]['sentence_index'] == chain[pi]['sentence_index'] and chain[pj]['start_index'] > chain[pi]['start_index']):
                        j = pj
                        i = pi
                    else:
                        j = pi
                        i = pj
                        
                    data_row = self.get_row(chain[i],chain[j])
                    if data_row != None:
                        x.append(data_row)
                        if 'is_coref' in chain[i]:
                            y.append(chain[i]['is_coref'])
                            raw_data.append([self.get_raw_data(chain[i],['chain_index','link_index'],[row_id,0]),\
                                         self.get_raw_data(chain[j],['chain_index','link_index'],[row_id,1])])
                    row_id = row_id + 1
        
        return x,y, raw_data
# data format
# is_coref doc_id chain_index link_index sentence_index
#pos_tag start_word_index end_word_index
#caevo_event ontonotes_event wordnet_event sentence                                                                                                                                                           

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

            the_rest = " ".join(tokens[8:])
            the_rest = the_rest.split("\t")
            sentence = []
            ss_tags = []
            
            for word in the_rest:
                t = word.split(" ")
                if len(t) != 3:
                    print("error")
                sentence.append(t[0])
                ss_tags.append(t[2])
              
            mention = {"raw_data":line,"pos_tag":pos_tag, \
                       "sentence_index": sentence_index, \
                       "start_index":start_index, \
                       "end_index":end_index, "sentence":sentence, \
                       "ss_tag":ss_tags,\
                       "is_coref":is_coref,"doc_id":doc_id}
            
            if prev_chain == None:
                    prev_chain = chain_index
            if prev_chain == chain_index:
                this_chain.append(mention)
            else:
                chains.append(this_chain)
                this_chain = [mention]
                prev_chain = chain_index
                
        except:
            print("line_skipped ", line)
    chains.append(this_chain)          
    return chains

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            np.random.seed(0)
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
