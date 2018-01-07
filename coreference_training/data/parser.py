testdocs= [93, 894, 56, 501, 693, 521, 743, 495, 625, 851, 620, 302, \
           439, 710, 87, 386, 868, 81, 296, 591, 74, 268, 937, 905, 570, \
           911, 674, 90, 424, 696, 28, 472, 145, 225, 260, 920, 568, 604, \
           357, 314, \
           274, 63, 658, 543, 609, 616, 517, 819, 166, 581, 846, 392]

def read_file(file_name):
    with open(file_name + ".txt","r",encoding="utf-8") as f:
        lines = f.readlines()
    lesk_lines = []
    with open("lesk\\"+file_name+"_lesk.txt",encoding="utf-8") as f:
        lesk_lines = f.readlines()
        
    chains = []
    this_chain = []
    prev_chain = None
    for (line,lesk_line) in zip(lines,lesk_lines):
        #try:
            tokens = line.split(" ")
            is_coref = int(tokens[0])
            doc_id = int(tokens[1])
            chain_index = int(tokens[2]) 
            link_index = int(tokens[3]) 
            sentence_index = int(tokens[4])
            pos_tag = tokens[5]
            start_index = int(tokens[6]) 
            end_index = int(tokens[7])
            #caevo_event = int(tokens[8])
            #ontonotes_event = int(tokens[9])
            #wordnet_event = int(tokens[10])
            sentence = tokens[8:]
            sent_ss = eval(lesk_line.replace("Synset('","'").replace("))",")"))
            if len(sent_ss) != len(sentence):
                print("ERROR ", sentence, sent_ss)
            new_sent = ""
            for (w1,w2) in zip(sentence,sent_ss):
                new_sent = new_sent + w1 + " " + str(w2[1]) + "\t"
                
            mention = {"raw_data":line,"pos_tag":pos_tag, \
                       "sentence_index": sentence_index, \
                       "start_index":start_index, \
                       "end_index":end_index, "sentence":sent_ss, \
                       "is_coref":is_coref,"doc_id":doc_id \
                       ,"caevo_event":0, "ontonotes_event":0,\
                       "wordnet_event":0,"chain_index":chain_index,\
                       "link_index":link_index}
            
            chains.append(mention)        
        #except:
         #   print("line_skipped")
                  
    return chains

def mark_sentence(s):
    x = s['sentence'].copy()
    x[s['start_index']] = "["+x[s['start_index']]
    x[s['end_index']] = x[s['end_index']] +"]"
    return " ".join(x)

def write_to_file(path, chains):
    output_file = open(path,"w+",encoding="utf-8")
    i = 0
    for link in chains:
        
        output_file.write(str(link['is_coref']) + " " + \
                          str(link['doc_id']) + " " + \
                          str(link['chain_index']) + " " + \
                          str(link['link_index']) + " " + \
                          str(link['sentence_index']) + " " + \
                          link['pos_tag'] + " " + \
                          str(link['start_index']) + " " + \
                          str(link['end_index']) + " " + \
                          #str(link['caevo_event']) + " " + \
                          #str(link['ontonotes_event']) + " " + \
                          #str(link['wordnet_event']) + " " + \
                          "\t".join(link['sentence']))
        '''
        output_file.write("S-"+str(i) + "\t"\
                          +"\t".join(link['sentence']).strip() + "\n")
        '''
        i = i + 1
    output_file.close()
