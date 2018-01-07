#SAME AS DATA_UTILS_MIXED but with updated gen features function
import io
import os
import numpy as np
import pickle
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.parsing.preprocessing import STOPWORDS
from textblob import Word
from textblob.wordnet import Synset
import nltk
import spacy
import wn_pos_conversion as wnconvert

#buckets for wordnet similarity and cosine similarity
#the buckets are created manually in a preprocessing step
# the last bucket is for unknown similarities 
cosine_bucket = [-1, 0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.9, 1.0, None]
wordnet_bucket = [0.08,0.1,0.12,0.14,0.16,0.18,0.2,0.26,0.34,0.5,1, None]
pos_tagset = []

with io.open('data/upenn_tagset.txt','r',encoding='utf-8') as f:
    pos_tagset = f.read().split()

class TextLoader():
    
    def __init__(self,sentences = None ,window = None,model_path = 'data\enwiki_stopwords.model',tensor_path=None, demo=False):
        # this is for testing purposes
        if sentences == None:
            self.load_model(model_path)
            return
        
        self.window = window
        self.nlp = spacy.load('en')
        self.demo = demo
        if demo:
             self.load_model(model_path)
             print("Generating tensors for demo")
             self.features,self.labels,self.raw_data = self.gen_features_and_labels(sentences) 

        else:
         if tensor_path is not None and os.path.exists(tensor_path):
            print("Loading saved tensors...")
            self.load_tensors(tensor_path)
            
         else:
          self.load_model(model_path)
         
          print("Generating tensors...")
          
          self.features,self.labels,self.raw_data = {},{},{}
          self.features['coref'],self.labels['coref'],self.raw_data['coref'] = self.gen_features_and_labels(sentences['coref'])
          self.features['incoref'],self.labels['incoref'],self.raw_data['incoref'] = self.gen_features_and_labels(sentences['incoref'])
          if tensor_path is not None:
            self.save_tensors(tensor_path)
                
        self.features_size = ( 2 * (2*self.window+1) * self.model_shape +  # size of the window (Contextual features)
                               2 * len(pos_tagset) + 
                               #2 * 3 + #event information (Event Features)
                               #2 +  # distance between sentences (Relational Features) and bit for doc-id equality in pair
                               1 * len(cosine_bucket) + # hotvector of mention headword similarity and avg mention similarity (Relational Features)
                               3 * len(wordnet_bucket) ) # hotvector of mention headword similarity and hypernyms (Relational Features)

    #Loading the word2vec model
    def load_model(self, model_path):
        print("Loading word2vec model...")
        word2vec_model = word2vec.Word2Vec.load(model_path)
        #word2vec_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        self.model = word2vec_model
        self.model_shape = self.model['news'].shape[0]      

    def save_tensors(self, tensors_path):
        tensors = [self.features, self.labels,self.model_shape,self.raw_data]
        with io.open(tensors_path, 'wb') as f:
            pickle.dump(tensors, f)


    def load_tensors(self, tensors_path):
        with io.open(tensors_path, 'rb') as f:
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
            
    #Intended for similarity of the mentions average
    def cosine_similarity(self,a,b):
        return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    
    #wordnet path similarity is not an equivalence relation
    def wordnet_path_sim(self,s1,s2):
        sim1 = s1.path_similarity(s2)
        sim2 = s2.path_similarity(s1)
        
        if sim1 == None or (sim2 != None and sim2 > sim1):
            return sim2
        else:
            return sim1
        
    def get_max_synsets(self,synsets1,synsets2):
        sim_max = None
        max_syn1 = None
        max_syn2 = None
        for synset1 in synsets1:
            for synset2 in synsets2:
                sim = self.wordnet_path_sim(synset1,synset2) 
                if sim_max == None or (sim != None and sim > sim_max):
                    sim_max = sim
                    max_syn1 = synset1
                    max_syn2 = synset2
                    
        return sim_max,max_syn1,max_syn2

    def convert_to_verb(self,max_syn,mention):
        if max_syn is not None and max_syn.pos() != 'v':
            a1 = []
            for m in mention:
                x = wnconvert.convert(m,'n','v')
                if len(x) < 1:
                    a1.append(m)
                else:
                    a1.append(x[0][0])
            return a1
        else:
            return mention
    
    def get_wordnet_sim(self,m1,m2,all_sim = True):
        a1,a2 = [],[]
        for m in m1:
            a1.append(Word(m).lemmatize("v"))
        for m in m2:
            a2.append(Word(m).lemmatize("v"))
        
        syn1 = Word("_".join(a1)).synsets
        syn2 = Word("_".join(a2)).synsets

        if len(syn1) < 1:
           for m in m1:
               syn1.extend(Word(m).synsets)
               
        if len(syn2) < 1:
           for m in m2:
               syn2.extend(Word(m).synsets)
    
        sim_max , max_syn1 , max_syn2 = self.get_max_synsets(syn1,syn2)
        hyper_sim = None
        verb_convert_sim = None
        if all_sim:
            if sim_max != None:
                hyp1,_ = zip(*max_syn1.hypernym_distances())
                hyp2,_ = zip(*max_syn2.hypernym_distances())
                hyper_sim, _, _ = self.get_max_synsets(list(hyp1),list(hyp2))
            # new similarity measure by converting everything to a verb
            a1,a2 = self.convert_to_verb(max_syn1,m1),self.convert_to_verb(max_syn2,m2)
            verb_convert_sim,_,_ = self.get_wordnet_sim(a1,a2,False)
                
        return sim_max,hyper_sim, verb_convert_sim
    
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
            
    def get_mention(self,sent_dict):
        s = sent_dict['start_index']
        e = sent_dict['end_index']
        return sent_dict['sentence'][s:e+1]
    
    # generates the embeddings of the the mention and the window arounds it
    # generates the head word
    # return the embeddings of the window.
    # If none of the words of the mention appear in the model, it returns none (can't be computed)
    def contextual_features(self,sent_dict):
        sentence = sent_dict['sentence']
        s = sent_dict['start_index']
        e = sent_dict['end_index']
        sentence = [x.replace('\n','')for x in sentence]
        row = []
        before_mention = sentence[:s]
        after_mention = sentence[e+1:]
        s = sent_dict['start_index']
        e = sent_dict['end_index']
        mention = sentence[s:e+1]
        #self.get_mention(sent_dict)
        
        #getting the headword
        doc = self.nlp(" ".join(mention))
        sents = list(doc.sents)
        try:
            head_word = str(sents[0].root).lower()
            head_word = (Word(head_word))
            pos_tag = sents[0].root.tag_

        except:
            #print 'Can\'t be encoded into string', sents[0].root
            head_word = sents[0].root
            head_word = (Word(head_word))
            pos_tag = sents[0].root.tag_
        #if unicode(sents[0].root) not in mention:
           #print 'Inconsistent tokenization for headword', sents[0].root,' in mention', mention, 'in sentence', sentence   
        
         #.stem()
        
        row.extend(self.k_words_embeddings(reversed(before_mention),self.window))
        m_avg = self.wordlist_average(mention)
        
        if not self.demo and m_avg == None:
            return None,None,None
        if self.demo and m_avg == None:
            return None,None,None   
        else:
            #Adding the mention's average embeddings
            row.extend(m_avg)
           
            #Adding the headword embedding
            #row.extend(self.words_vec([head_word],1))
            row.extend(self.k_words_embeddings(after_mention,self.window))

            postag_row = [0] * len(pos_tagset)
            m_pt = nltk.pos_tag(mention)
            for w,tag in m_pt:
                pi = pos_tagset.index(tag)
                postag_row[pi] = postag_row[pi] + 1
                
            row.extend(postag_row)
            return row, head_word, m_avg

    # creates one row of input for the neural network
    # return None, if the none of the words in the mention are in the word2vec model
    def get_row(self,sentence_i, sentence_j):
        data_row = []            
        # appending the vecotrs for the chosen window for both sentences
        data_row_i, mention_i, avg_i = self.contextual_features(sentence_i)
        data_row_j ,mention_j, avg_j = self.contextual_features(sentence_j)
        if avg_i == None or avg_j == None:
            return None
        
        else:   
            # appending the vecotrs for the chosen window for sentences_i
            data_row.extend(data_row_i)
                             
            #repeat for j
            data_row.extend(data_row_j)
            
            # adds any additional features
            #       feature 1: distance between senteces 
            #data_row.append(sentence_j['sentence_index'] - sentence_i['sentence_index'])

            #data_row.append(int(sentence_j['doc_id']==sentence_i['doc_id']))
            
            #       feature 2: cosine sim of headwords of mentions
            cosine_sim = self.get_word2vec_sim(mention_i,mention_j)
            data_row.extend(self.get_bucket(cosine_bucket, cosine_sim))
            
            #       feature 3: cosine sim of average mentions
            #avg_cosine_sim = self.cosine_similarity(avg_i,avg_j)
            #data_row.extend(self.get_bucket(cosine_bucket, avg_cosine_sim))
            #       feature 4 & 5 : wordnet sim of mentions and their hypernyms
            wordnet_sim,wordnet_hypersim,wordnet_verbsim = self.get_wordnet_sim(\
                self.get_mention(sentence_i),\
                self.get_mention(sentence_j))
            data_row.extend(self.get_bucket(wordnet_bucket, wordnet_sim))
            data_row.extend(self.get_bucket(wordnet_bucket, wordnet_hypersim))
            data_row.extend(self.get_bucket(wordnet_bucket, wordnet_verbsim))
            
            return data_row
        
    # The following is used to generate the 'raw_data'
    def get_raw_data(self,link):
        mention = link.copy()
        #for k,v in zip(keys,values):
            #mention[k] = v
        mention['raw_data'] = self.get_mention_line(mention)
        return mention
        
    def get_mention_line(self,mention):
        data = ["is_coref","split_id","doc_id","chain_index","link_index", \
                "sentence_index","pos_tag","start_index","end_index"]

        mention_s = ""
        for key in data:
           try:
                mention_s = mention_s + str(mention[key]) + " "
           except:
             continue 
            
        mention_s = mention_s + " ".join(mention["sentence"])
        return mention_s.rstrip()
    
    #expects a list of pairs - each pair forms a row
    def gen_features_and_labels(self,pairlist,data=""):
        x, y, raw_data, row_id = [], [], [], 0
        for pair in pairlist:
            data_row = self.get_row(pair[0],pair[1])
            if data_row != None:
                x.append(data_row)
                y.append(pair[0]['is_coref'])
                mention1 = self.get_raw_data(pair[0])
                mention2 = self.get_raw_data(pair[1])
                raw_data.append([mention1, mention2])
            row_id = row_id + 1
        return x,y, raw_data
