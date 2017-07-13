import sys
import utils
import numpy as np
import tensorflow as tf
from config import Config

# Source : https://gist.github.com/harpone/3453185b41d8d985356cbe5e57d67342
def py_func(func, inp, Tout, stateful=True, name=None, grad=None):
    rnd_name = 'PyFuncGrad' + str(np.random.randint(0, 1E+8))
    tf.RegisterGradient(rnd_name)(grad)  
    g = tf.get_default_graph()
    with g.gradient_override_map({"PyFunc": rnd_name}):
        return tf.py_func(func, inp, Tout, stateful=stateful, name=name)

class Network(object):
    def __init__(self, config, summarizer):
        tf.set_random_seed(1234)
        self.summarizer = summarizer
        self.config = config
        self.Wx1, self.Wx2, self.Wx3, self.bx1, self.bx2, self.bx3 = self.init_Fx_variables()
        self.We1, self.We2, self.be1, self.be2 = self.init_Fe_variables()
        self.Wd1, self.Wd2, self.bd1, self.bd2 = self.init_Fd_variables()

    def weight_variable(self, shape, name):
        return tf.get_variable(name=name, shape=shape,  initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, shape, name):
        return tf.Variable(tf.constant(0.1, shape=shape), name=name)

    def init_Fx_variables(self):
        W1 = self.weight_variable([self.config.features_dim, self.config.solver.hidden_dim], "weight_x1")
        W2 = self.weight_variable([self.config.solver.hidden_dim, self.config.solver.hidden_dim], "weight_x2")
        W3 = self.weight_variable([self.config.solver.hidden_dim, self.config.solver.latent_embedding_dim], "weight_x3")
        b1 = self.bias_variable([self.config.solver.hidden_dim], "bias_x1")
        b2 = self.bias_variable([self.config.solver.hidden_dim], "bias_x2")
        b3 = self.bias_variable([self.config.solver.latent_embedding_dim], "bias_x3")
        return W1, W2, W3, b1, b2, b3

    def init_Fe_variables(self):
        W1 = self.weight_variable([self.config.labels_dim, self.config.solver.hidden_dim], "weight_e1")
        W2 = self.weight_variable([self.config.solver.hidden_dim, self.config.solver.latent_embedding_dim], "weight_e2")
        b1 = self.bias_variable([self.config.solver.hidden_dim], "bias_e1")
        b2 = self.bias_variable([self.config.solver.latent_embedding_dim], "bias_e2") 
        return W1, W2, b1, b2

    def init_Fd_variables(self):
        W1 = self.weight_variable([self.config.solver.latent_embedding_dim, self.config.solver.hidden_dim], "weight_d1")
        W2 = self.weight_variable([self.config.solver.hidden_dim, self.config.labels_dim], "weight_d2")
        b1 = self.bias_variable([self.config.solver.hidden_dim], "bias_d1")
        b2 = self.bias_variable([self.config.labels_dim], "bias_d2")
        return W1, W2, b1, b2

    def accuracy(self, y_pred, y):
        return tf.reduce_mean(tf.cast(tf.equal(tf.round(y_pred), y), tf.float32))

    def Fx(self, X, keep_prob):
        hidden1 = tf.nn.dropout(utils.leaky_relu(tf.matmul(X, self.Wx1) + self.bx1), keep_prob)
        hidden2 = tf.nn.dropout(utils.leaky_relu(tf.matmul(hidden1, self.Wx2) + self.bx2), keep_prob)
        hidden3 = tf.nn.dropout(utils.leaky_relu(tf.matmul(hidden2, self.Wx3) + self.bx3), keep_prob)
        return hidden3

    def Fe(self, Y, keep_prob):
        hidden1 = tf.nn.dropout(utils.leaky_relu(tf.matmul(Y, self.We1)) + self.be1, keep_prob)
        pred = tf.nn.dropout(utils.leaky_relu(tf.matmul(hidden1, self.We2) + self.be2), keep_prob)
        return pred

    def Fd(self, input, keep_prob):
        hidden1 = tf.nn.dropout(utils.leaky_relu(tf.matmul(input, self.Wd1) + self.bd1), keep_prob)
        y_pred = tf.matmul(hidden1, self.Wd2) + self.bd2
        return y_pred

    def prediction(self, X, keep_prob):
        Fx = self.Fx(X, keep_prob)
        return self.Fd(Fx, keep_prob)

    def embedding_loss(self, Fx, Fe):
        Ix, Ie = tf.eye(tf.shape(Fx)[0]), tf.eye(tf.shape(Fe)[0])
        C1, C2, C3 = tf.abs(Fx - Fe), tf.matmul(Fx, tf.transpose(Fx)) - Ix, tf.matmul(Fe, tf.transpose(Fe)) - Ie
        return tf.reduce_mean(tf.square(Fx - Fe)) #tf.trace(tf.matmul(C1, tf.transpose(C1))) + self.config.solver.lagrange_const * tf.trace(tf.matmul(C2, tf.transpose(C2))) + self.config.solver.lagrange_const * tf.trace(tf.matmul(C3, tf.transpose(C3)))

    # My blood, sweat and tears were also embedded into the emebedding.
    def output_loss(self, predictions, labels):
        Ei = 0.0
        i, cond = 0, 1
        while cond == 1:
            cond = tf.cond(i >= tf.shape(labels)[0] - 1, lambda: 0, lambda: 1)
            prediction_, Y_ = tf.slice(predictions, [i, 0], [1, self.config.labels_dim]), tf.slice(labels, [i, 0], [1, self.config.labels_dim])
            zero, one = tf.constant(0, dtype=tf.float32), tf.constant(1, dtype=tf.float32)
            ones, zeros = tf.gather_nd(prediction_, tf.where(tf.equal(Y_, one))), tf.gather_nd(prediction_, tf.where(tf.equal(Y_, zero)))
            y1 = tf.reduce_sum(Y_)
            y0 = Y_.get_shape().as_list()[1] - y1
            temp = (1/y1 * y0) * tf.reduce_sum(tf.exp(-(tf.reduce_sum(ones) / tf.cast(tf.shape(ones)[0], tf.float32) - tf.reduce_sum(zeros) / tf.cast(tf.shape(zeros)[0], tf.float32))))
            Ei += tf.cond(tf.logical_or(tf.is_inf(temp), tf.is_nan(temp)), lambda : tf.constant(0.0), lambda : temp)
            i += 1
        return Ei 

    def cross_loss(self, features, labels, keep_prob):
        predictions = self.prediction(features, keep_prob)
        Fx = self.Fx(features, keep_prob)
        Fe = self.Fe(labels, keep_prob)
        cross_loss = tf.add(tf.log(1e-10 + tf.nn.sigmoid(predictions)) * labels, tf.log(1e-10 + (1 - tf.nn.sigmoid(predictions))) * (1 - labels))
        cross_entropy_label = -1 * tf.reduce_mean(tf.reduce_sum(cross_loss, 1))
        return cross_entropy_label

    def loss(self, features, labels, keep_prob):
        lamda = 0.00
        prediction = tf.nn.sigmoid(self.prediction(features, keep_prob))
        Fx = self.Fx(features, keep_prob)
        Fe = self.Fe(labels, keep_prob)
        l2_norm = tf.reduce_sum(tf.square(self.Wx1)) + tf.reduce_sum(tf.square(self.Wx2)) + tf.reduce_sum(tf.square(self.Wx3)) + tf.reduce_sum(tf.square(self.We1)) + tf.reduce_sum(tf.square(self.We2)) + tf.reduce_sum(tf.square(self.Wd1)) + tf.reduce_sum(tf.square(self.Wd2))
        return self.embedding_loss(Fx, Fe) + self.config.solver.alpha * self.output_loss(prediction, labels) + lamda * l2_norm # self.cross_loss(features, labels, keep_prob) 

    def train_step(self, loss):
        optimizer = self.config.solver.optimizer
        return optimizer(self.config.solver.learning_rate).minimize(loss)
