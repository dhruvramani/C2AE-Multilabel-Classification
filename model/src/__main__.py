import os
import sys
import time
import utils
import numpy as np
import tensorflow as tf
from parser import Parser
from config import Config
from network import Network
from dataset import DataSet
from eval_performance import evaluate, patk
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class Model(object):
    def __init__(self, config):
        self.epoch_count = 0
        self.config = config
        self.data = DataSet(config)
        self.add_placeholders()
        self.summarizer = tf.summary
        self.net = Network(config, self.summarizer)
        self.optimizer = self.config.solver.optimizer
        self.y_pred = self.net.prediction(self.x, self.keep_prob)
        self.loss = self.net.loss(self.x, self.y, self.keep_prob)
        self.accuracy = self.net.accuracy(tf.nn.sigmoid(self.y_pred), self.y)
        self.summarizer.scalar("accuracy", self.accuracy)
        self.summarizer.scalar("loss", self.loss)
        self.train = self.net.train_step(self.loss)
        self.saver = tf.train.Saver()
        self.init = tf.global_variables_initializer()

    def add_placeholders(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.config.features_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, self.config.labels_dim])
        self.keep_prob = tf.placeholder(tf.float32)

    def run_epoch(self, sess, data, summarizer, epoch):
        err = list()
        i = 0
        step = epoch
        merged_summary = self.summarizer.merge_all()
        for X, Y, tot in self.data.next_batch(data):
            feed_dict = {self.x : X, self.y : Y, self.keep_prob : self.config.solver.dropout}
            if not self.config.load:
                summ, _, y_pred, loss = sess.run([merged_summary, self.train, self.y_pred, self.loss], feed_dict=feed_dict)
                err.append(loss) 
                output = "Epoch ({}) Batch({}) - Loss : {}".format(self.epoch_count, i, loss)
                with open("../stdout/{}_train.log".format(self.config.project_name), "a+") as log:
                    log.write(output + "\n")
                print("   {}".format(output), end='\r')
            step = int(epoch*tot + i)
            summarizer.add_summary(summ, step)
            i += 1
        return np.mean(err), step

    def run_eval(self, sess, data, summary_writer=None, step=0):
        y, y_pred, loss_, metrics, p_k = list(), list(), 0.0, None, None
        accuracy, loss = 0.0, 0.0
        merged_summary = self.summarizer.merge_all()
        i = 0
        for X, Y, tot in self.data.next_batch(data):
            feed_dict = {self.x: X, self.y: Y, self.keep_prob: 1}
            if i == tot-1 and summary_writer is not None:
                if data == "validation":
                    summ, loss_ =  sess.run([merged_summary, self.loss], feed_dict=feed_dict)
                else :
                    summ, loss_, accuracy_val = sess.run([merged_summary, self.loss, self.accuracy], feed_dict=feed_dict)
                summary_writer.add_summary(summ, step)
            else:
                if data == "validation":
                    loss_, Y_pred=  sess.run([self.loss, tf.nn.sigmoid(self.y_pred)], feed_dict=feed_dict)
                    p_k = patk(predictions=Y_pred, labels=Y)
                else :
                    loss_, Y_pred, accuracy_val = sess.run([self.loss, tf.nn.sigmoid(self.y_pred), self.accuracy], feed_dict=feed_dict)
                    metrics = evaluate(predictions=Y_pred, labels=Y)
                    p_k = patk(predictions=Y_pred, labels=Y)
                    accuracy += accuracy_val #metrics['accuracy']
            loss += loss_
            i += 1
        return loss / i , accuracy / self.config.batch_size, metrics, p_k
    
    def add_summaries(self, sess):
        if self.config.load or self.config.debug:
            path_ = "../results/tensorboard"
        else :
            path_ = "../bin/results/tensorboard"
        summary_writer_train = tf.summary.FileWriter(path_ + "/train", sess.graph)
        summary_writer_val = tf.summary.FileWriter(path_ + "/val", sess.graph)
        summary_writer_test = tf.summary.FileWriter(path_+ "/test", sess.graph)
        summary_writers = {'train': summary_writer_train, 'val': summary_writer_val, 'test': summary_writer_test}
        return summary_writers

    def fit(self, sess, summarizer):
        '''
         - Patience Method : 
         + Train for particular no. of epochs, and based on the frequency, evaluate the model using validation data.
         + If Validation Loss increases, decrease the patience counter.
         + If patience becomes less than a certain threshold, devide learning rate by 10 and switch back to old model
         + If learning rate is lesser than a certain 
        '''
        max_epochs = self.config.max_epochs
        patience = self.config.patience
        patience_increase = self.config.patience_increase
        improvement_threshold = self.config.improvement_threshold
        best_validation_loss = 1e6
        self.epoch_count = 0
        best_step, losses, learning_rate = -1, list(), self.config.solver.learning_rate
        while self.epoch_count < max_epochs :
            if(self.config.load == True):
                break
            start_time = time.time()
            average_loss, tr_step = self.run_epoch(sess, "train", summarizer['train'], self.epoch_count)
            duration = time.time() - start_time
            if not self.config.debug :
                if self.epoch_count % self.config.epoch_freq == 0 :
                    val_loss, _, _, _ = self.run_eval(sess, "validation", summarizer['val'], tr_step)
                    test_loss, _, metrics, _= self.run_eval(sess, "test", summarizer['test'], tr_step)
                    output =  "=> Training : Loss = {:.2f} | Validation : Loss = {:.2f} | Test : Loss = {:.2f}".format(average_loss, val_loss, test_loss)
                    with open("../stdout/validation.log", "a+") as f:
                        output_ = output + "\n=> Test : Coverage = {}, Average Precision = {}, Micro Precision = {}, Micro Recall = {}, Micro F Score = {}".format(metrics['coverage'], metrics['average_precision'], metrics['micro_precision'], metrics['micro_recall'], metrics['micro_f1'])
                        output_ += "\n=> Test : Macro Precision = {}, Macro Recall = {}, Macro F Score = {}\n\n\n".format(metrics['macro_precision'], metrics['macro_recall'], metrics['macro_f1'])
                        f.write(output_)
                    print(output)
                    if self.config.have_patience:
                        if val_loss < best_validation_loss :
                            if val_loss < best_validation_loss * improvement_threshold :
                                self.saver.save(sess, self.config.ckptdir_path + "model_best.ckpt")
                                best_validation_loss = val_loss
                                best_step = self.epoch_count
                        else :
                            if patience < 1:
                                self.saver.restore(sess, self.config.ckptdir_path + "model_best.ckpt")
                                if learning_rate <= 0.00001 :
                                    print("=> Breaking by Patience Method")
                                    break
                                else :
                                    learning_rate /= 10
                                    patience = self.config.patience
                                    print("\033[91m=> Learning rate dropped to {}\033[0m".format(learning_rate))
                            else :
                                patience -= 1
            self.epoch_count += 1
        print("=> Best epoch : {}".format(best_step))
        if self.config.debug == True:
            sys.exit()
        test_loss, test_accuracy, test_metrics, p_k = self.run_eval(sess, "test", summarizer['test'], tr_step)
        returnDict = {"test_loss" : test_loss, "test_accuracy" : test_accuracy, 'test_metrics' : test_metrics, "test_pak" : p_k}
        if self.config.debug == False:
            returnDict["train"] =  best_validation_loss
        return returnDict


def init_model(config):
    tf.reset_default_graph()
    tf.set_random_seed(1234)
    with tf.variable_scope('Model', reuse=None) as scope:
        model = Model(config)
    tf_config = tf.ConfigProto(allow_soft_placement=True)#, device_count = {'GPU': 0})
    tf_config.gpu_options.allow_growth = True
    sm = tf.train.SessionManager()

    if config.retrain or config.load == True:
        print("=> Loading model from checkpoint")
        load_ckpt_dir = config.ckptdir_path
    else:
        print("=> No model loaded from checkpoint")
        load_ckpt_dir = ''
    sess = sm.prepare_session("", init_op=model.init, saver=model.saver, checkpoint_dir=load_ckpt_dir, config=tf_config)
    if config.load == True :
        saver = tf.train.Saver()
        sess_ = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
        saver.restore(sess_, config.ckptdir_path + "/resultsmodel_best.ckpt")
        return model, sess
    return model, sess

def train_model(config):
    if config.load == True:
        print("\033[92m=>\033[0m Testing Model")
    else: 
        print("\033[92m=>\033[0m Training Model")
    model, sess = init_model(config)
    with sess:
        summary_writers = model.add_summaries(sess)
        loss_dict = model.fit(sess, summary_writers)
        return loss_dict

def main():
    args = Parser().get_parser().parse_args()
    config = Config(args)
    loss_dict = train_model(config)
    metrics = loss_dict['test_metrics']
    if config.debug == False:
        output = "=> Best Train Loss : {}, Test Loss : {}, Test Accuracy : {}".format(loss_dict["train"], loss_dict["test_loss"], loss_dict["test_accuracy"])
    else : 
        output = "=> Test Loss : {}, Test Accuracy : {}".format(loss_dict["test_loss"], loss_dict["test_accuracy"])
    output += "\n=> Test : Coverage = {}, Average Precision = {}, Micro Precision = {}, Micro Recall = {}, Micro F Score = {}".format(metrics['coverage'], metrics['average_precision'], metrics['micro_precision'], metrics['micro_recall'], metrics['micro_f1'])
    output += "\n=> Test : Macro Precision = {}, Macro Recall = {}, Macro F Score = {}".format(metrics['macro_precision'], metrics['macro_recall'], metrics['macro_f1'])
    output += "\n=> Test : p@K values : {}".format(loss_dict['test_pak'])
    with open("../stdout/test_log.log", "a+") as f:
        f.write(output)
    print("\033[1m\033[92m{}\033[0m\033[0m".format(output))

if __name__ == '__main__' :
    np.random.seed(1234)
    main() # Phew!