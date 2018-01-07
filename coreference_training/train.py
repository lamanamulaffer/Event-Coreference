import tensorflow as tf
import os
import argparse
import data_utils as data_utils
import model_00
import model_01
import time
import random
from gensim.models import word2vec
from gensim.models.keyedvectors import KeyedVectors

CHECKPOINT_DIR = 'checkpoints'
LOGS_DIR = 'logs'
CACHE_DIR = 'cache'
# Evaluate model on training set every this number of steps
EVALUATE_EVERY = 100
# Save a checkpoint every this number of steps
CHECKPOINT_EVERY = 100

# offset: intended for k-fold validation
def load_data(coref_data,incoref_data, window, model_path, testing_count, batch_size, n_epochs, offset = 0,  write_testset = False):
   
    with open(coref_data,'r', encoding="utf8",errors="replace") as f:
        sentences_coref = f.readlines()
        
    with open(incoref_data,'r', encoding="utf8",errors="replace") as f:
        sentences_incoref = f.readlines()
        
    tensor_path = os.path.join(CACHE_DIR, 'tensors.pkl')
    
    print("Reading data from: ",coref_data," and ", incoref_data )
    sentences = {'coref': data_utils.read_file(sentences_coref),'incoref':data_utils.read_file(sentences_incoref)}
    
    textloader = data_utils.TextLoader(sentences, window, model_path, tensor_path)
    x_coref = textloader.features['coref']
    y_coref = textloader.labels['coref']
    xt_incoref = textloader.features['incoref']
    yt_incoref = textloader.labels['incoref']
    raw_data_coref = textloader.raw_data['coref']
    raw_datat_incoref = textloader.raw_data['incoref']
    
    x_incoref,y_incoref,raw_data_incoref = [],[],[]
    #randomize incoref_data
    new_ids = list(range(len(xt_incoref)))
    random.seed(100)
    random.shuffle(new_ids)

    for i in new_ids:
        x_incoref.append(xt_incoref[i])
        y_incoref.append(yt_incoref[i])
        raw_data_incoref.append(raw_datat_incoref[i])
        
    
    print("--------------- DATA STATS --------------- ")
    print("Total extracted positive data: ", len(y_coref))
    print("Total extracted negative data: ", len(y_incoref))
    print()
    # Seperation into testing and training
    #counting the number of coref data in each doc
    docs_count = {}
    max_doc_id = 0 
    for m in raw_data_coref:
        doc_id = m[0]['doc_id']
        if doc_id in docs_count:
            docs_count[doc_id] = docs_count[doc_id] + 1
        else:
            docs_count[doc_id] = 1
        if doc_id > max_doc_id:
            max_doc_id = doc_id
            
    # shuffling according to the number of docs
    docs_seq_ind = list(range(max_doc_id))
    random.seed(400)
    random.shuffle(docs_seq_ind)
    
    # to get the number of test cases
    count = 0
    offset_c = 0
    start = 0
    for x in range(len(docs_seq_ind)):
        if count + 10 >= testing_count:
            if offset_c >= offset: # allowing a window of 10 chains
                print("testing documents:", docs_seq_ind[start:x])
                break;
            else:
                count = 0
                start = x
                offset_c = offset_c + 1
        if docs_seq_ind[x] in docs_count:
            count = count + docs_count[docs_seq_ind[x]]
    
    docs_seq_ind = docs_seq_ind[start:x]
    
    x_test,x_train,y_test,y_train = [],[],[],[]
    raw_data = []
    for (x,y,r) in zip(x_coref,y_coref,raw_data_coref):
        if r[0]['doc_id'] in docs_seq_ind:
            x_test.append(x)
            y_test.append(y)
            raw_data.append(r)
        else:
            x_train.append(x)
            y_train.append(y)
    coref_test_size = len(x_test)
    coref_train_size = len(x_train)
    
    for (x,y,r) in zip(x_incoref,y_incoref,raw_data_incoref):
        if r[0]['doc_id'] in docs_seq_ind:
            if len(x_test) - coref_test_size < coref_test_size:
                x_test.append(x)
                y_test.append(y)
                raw_data.append(r)
        else:
            if len(x_train) - coref_train_size < coref_train_size:
                x_train.append(x)
                y_train.append(y)
    # End of Speration into test and training 
    if write_testset:
        with open("devset_" + str(testing_count) +"_" + \
                  coref_data.split("/")[1],"w+",encoding="utf-8") as f:
            for chain in raw_data:
                for link in chain:
                    f.write(link['raw_data']+"\n")
    
        
    train_batches = data_utils.batch_iter(
    list(zip(x_train, y_train)), batch_size, n_epochs)
    test_data = {'x': x_test, 'y': y_test}
    print()
    print("Total testing data: ", len(y_test), " coref:", coref_test_size)
    print("Total training data: ", len(y_train), " coref:", coref_train_size)
    print()
    
    return (train_batches, test_data, textloader.features_size)


def model_init(features_size, ncoref_classes, L1_size):
    global_step = tf.Variable(
        initial_value=0, name="global_step", trainable=False)
    if L1_size > 0 : 
        coref_model = model_01.CorefModel(features_size, ncoref_classes,L1_size)
    else:
        coref_model = model_00.CorefModel(features_size, ncoref_classes)
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(coref_model.loss, global_step=global_step)

    return coref_model, train_op, global_step

def logging_init(model, graph):
    """
    Set up logging so that progress can be visualised in TensorBoard.
    """
    # Add ops to record summaries for loss and accuracy...
    train_loss = tf.summary.scalar("train_loss", model.loss)
    train_accuracy = tf.summary.scalar("train_accuracy", model.accuracy)
    # ...then merge these ops into one single op so that they easily be run
    # together
    train_summary_ops = tf.summary.merge([train_loss, train_accuracy])
    # Same ops, but with different names, so that train/test results show up
    # separately in TensorBoard
    test_loss = tf.summary.scalar("test_loss", model.loss)
    test_accuracy = tf.summary.scalar("test_accuracy", model.accuracy)
    test_summary_ops = tf.summary.merge([test_loss, test_accuracy])

    timestamp = int(time.time())
    run_log_dir = os.path.join(LOGS_DIR, str(timestamp))
    os.makedirs(run_log_dir)
    # (this step also writes the graph to the events file so that
    # it shows up in TensorBoard)
    summary_writer = tf.summary.FileWriter(run_log_dir, graph)

    return train_summary_ops, test_summary_ops, summary_writer

def checkpointing_init():
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
    return saver


def step(sess, model, standard_ops, train_ops, test_ops, x, y, summary_writer,
         train):
    feed_dict = {model.input_x: x, model.input_y: y}

    if train:
        step, loss, accuracy, _, summaries = sess.run(standard_ops + train_ops,
                                                      feed_dict)
    else:
        step, loss, accuracy, summaries = sess.run(standard_ops + test_ops,
                                                   feed_dict)

    print("Step %d: loss %.1f, accuracy %d%%" % (step, loss, 100 * accuracy))
    summary_writer.add_summary(summaries, step)

def parse_args():
    parser = argparse.ArgumentParser()
    
    #intended for k-fold
    parser.add_argument(
        "--offset",
        type=int,
        default=7,
        help="offset for cross validation")

    parser.add_argument(
        "--L1_size",
        type=int,
        default=200,
        help="size of the hidden layer")
    
    parser.add_argument(
        "--write_testset",
        type=int,
        default=False,
        help="creating a file of the testset (dev)")

    parser.add_argument(
        "--coref_data",
        type=str,
        default="data/ecb_coref_train.txt",
        help="The set of co-referent data")
    
    parser.add_argument(
        "--incoref_data",
        type=str,
        default="data/ecb_incoref_train.txt",
        help="The set of inco-referent data")
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="data/enwiki_stopwords.model",
        help="The path for the word2vec model")

    parser.add_argument(
        "--window_size",
        type=int,
        default=2,
        help="The size of the window on either sides of the mention")
    
    parser.add_argument(
        "--testing_count",
        type=int,
        default=200,
        help="Number of items in the testset (devset) ")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="")
    
    args = parser.parse_args()
    return args


def main():

    
    n_epochs = 100
    ncoref_classes = 2
    args = parse_args()

    print("#args used:",args)
    train_batches, test_data,features_size = load_data(args.coref_data, args.incoref_data, args.window_size, \
                                                       args.model_path, args.testing_count, args.batch_size, n_epochs, \
                                                       args.offset,args.write_testset)
    x_test = test_data['x']
    y_test = test_data['y']

    
    sess = tf.Session()
    coref_model, train_op, global_step = model_init(features_size, ncoref_classes,args.L1_size)
    
    train_summary_ops, test_summary_ops, summary_writer = logging_init(
        coref_model, sess.graph)
    saver = checkpointing_init()

    sess.run(tf.global_variables_initializer())
    sess.graph.finalize()

    standard_ops = [global_step, coref_model.loss, coref_model.accuracy]
    train_ops = [train_op, train_summary_ops]
    test_ops = [test_summary_ops]

    for batch in train_batches:
        x_batch, y_batch = zip(*batch)
        step(
            sess,
            coref_model,
            standard_ops,
            train_ops,
            test_ops,
            x_batch,
            y_batch,
            summary_writer,
            train=True)
        current_step = tf.train.global_step(sess, global_step)

        if current_step % EVALUATE_EVERY == 0:
            print("\nEvaluation:")
            step(
                sess,
                coref_model,
                standard_ops,
                train_ops,
                test_ops,
                x_test,
                y_test,
                summary_writer,
                train=False)
            print("")

        if current_step % CHECKPOINT_EVERY == 0:
            prefix = os.path.join(CHECKPOINT_DIR, 'model')
            path = saver.save(sess, prefix, global_step=current_step)
            print("Saved model checkpoint to '%s'" % path)
    sess.close()
    
main()
