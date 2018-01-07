import tensorflow as tf

class CorefModel(object):
    def __init__(self,features_size, ncoref_classes):
        print("Inititalizing model")
        self.input_x = tf.placeholder(tf.float32,[None,features_size], name = "input_x")
        self.input_y = tf.placeholder(tf.int64, [None], name="input_y")
        
        # creating it with no hidden layers, is there a linear relationship?
        with tf.name_scope("model"):
            w1 = tf.Variable(tf.truncated_normal([features_size,ncoref_classes],stddev=0.1,seed=1))
            b = tf.Variable(tf.truncated_normal([ncoref_classes],seed=2))
            self.logits = tf.matmul(self.input_x,w1) + b

            self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.input_y, logits=self.logits))

        with tf.name_scope("accuracy"):
            self.predictions = tf.argmax(self.logits,axis=1,name='predictions')
            correct_prediction = tf.equal(self.predictions, self.input_y)
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))
                                       
