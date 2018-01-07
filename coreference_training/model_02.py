import tensorflow as tf

class CorefModel(object):
    def __init__(self, features_size, ncoref_classes,h1_size=75,h2_size=25):
        print("Inititalizing model")
        self.input_x = tf.placeholder(tf.float32,[None,features_size], name = "input_x")
        self.input_y = tf.placeholder(tf.int64, [None], name="input_y")
        
        with tf.name_scope("model"):
            
            #TWO layer neural network
            
            wi = tf.Variable(tf.truncated_normal([features_size,h1_size], stddev=0.1))
            
            self.h1 = tf.nn.relu(tf.matmul(self.input_x, wi))

            self.w1 = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.1))
   
            self.h2 = tf.nn.relu(tf.matmul(self.h1, self.w1))

            self.w2 = tf.Variable(tf.truncated_normal([h2_size, ncoref_classes], stddev=0.1))
                  
            self.logits = tf.matmul(self.h2,self.w2)

            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.input_y, logits=self.logits))
            
        with tf.name_scope("accuracy"):
            self.predictions = tf.argmax(self.logits,axis=1,name='predictions')
            correct_prediction = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))
                                       
