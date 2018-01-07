import tensorflow as tf

class CorefModel(object):
    def __init__(self, features_size, ncoref_classes,h_size=75):
        print("Inititalizing model")
        self.input_x = tf.placeholder(tf.float32,[None,features_size], name = "input_x")
        self.input_y = tf.placeholder(tf.int64, [None], name="input_y")
        
        with tf.name_scope("model"):
            #One layer neural network
            
            w1 = tf.Variable(tf.truncated_normal([features_size,h_size],stddev=0.1,seed=1))
            #w1 = tf.Variable(tf.fill([features_size,h_size],0.01))
            self.h = tf.nn.relu(tf.matmul(self.input_x, w1))
            self.w2 = tf.Variable(
                tf.truncated_normal([h_size, ncoref_classes], stddev=0.1,seed=2))
            #self.w2 = tf.Variable(
            #    tf.fill([h_size, ncoref_classes],0.01))
            self.logits = tf.matmul(self.h,self.w2)

            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.input_y, logits=self.logits))

        with tf.name_scope("accuracy"):
            dims = tf.stack([tf.shape(self.logits)[0],1])
            extra_row = tf.fill(dims,0.6)
            new_logits = tf.concat([self.logits,extra_row],axis=1)
            #new_logits = self.logits
            self.predictions = tf.argmax(new_logits \
                                                   ,axis=1,name='predictions')
            correct_prediction = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))
                                       
