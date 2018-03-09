from util import initia_data
from model import TDLSTMmodel
import random
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import os

# path for log, model and result
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("embeding_file", './glove.twitter.27B.100d.txt', "path for embeding files")
tf.app.flags.DEFINE_string("train_file", './train', "path to train file")
tf.app.flags.DEFINE_string("test_file", './test', "path to test file")
tf.app.flags.DEFINE_integer("max_article_len", 150, "max_article_len")
tf.app.flags.DEFINE_integer("embedingDim", 100, "word dimention")
tf.app.flags.DEFINE_integer("word_hidden_dim", 200, "lstm hidden dimention")
tf.app.flags.DEFINE_integer("max_target_len", 5, "max_target_len")
tf.app.flags.DEFINE_integer("batch_size", 20, "batch_size")
tf.app.flags.DEFINE_integer("print_every", 20, "print_every")
tf.app.flags.DEFINE_float("clip", 5, "gradient to clip")
tf.app.flags.DEFINE_float("learning_rate", 0.05, "initial learning rate")
tf.app.flags.DEFINE_string("model_path", './weights', "path to save model")
tf.app.flags.DEFINE_integer("max_epoch", 200, "max_epoch")
tf.app.flags.DEFINE_string("train_summary_dir", './summary_ada', "summary")
tf.app.flags.DEFINE_float("max_gradient", 1.0, " max_gradient")

def main(_):
    train_dataset, test_dataset, embeding_file, vocab_dic = initia_data(FLAGS)
    model = TDLSTMmodel(embeding_file, FLAGS)
    
    best_test_f1 = 0
    with tf.Session() as sess:
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        train_summary_writer = tf.summary.FileWriter(FLAGS.train_summary_dir, sess.graph)
        sess.run(tf.global_variables_initializer())
        cost = 0.0
        for _ in range(FLAGS.max_epoch):
            steps_per_epoch = len(train_dataset) // FLAGS.batch_size + 1
            random.shuffle(train_dataset)
            print("begin")
            for batch in get_batch(train_dataset, FLAGS.batch_size):
                global_step = model.global_step.eval()
                step = global_step % steps_per_epoch
                loss, _, summary = model.runstep(sess, batch, True)
                
                cost += loss / FLAGS.print_every

                if global_step % FLAGS.print_every == 0:
                    print("loss: %f" % cost)
                    train_summary_writer.add_summary(summary, global_step)
                    cost = 0
                
                
            print("validating ner")
            predicts = []
            rightnums = []
            for batch in get_batch(test_dataset, FLAGS.batch_size):
                predict, rightnum = model.runstep(sess, batch, False)
                rightnums.append(rightnum)
                predicts.append(predict)
            #accuracy = np.mean(rightnums)
            sum = 0
            lens = 0
            for right_i in rightnums:
                sum = sum + np.sum(right_i)
                lens = lens +len(right_i)
            accuracy = sum / lens
            print("accuracy: %f" % accuracy)
            if accuracy > best_test_f1:
                best_test_f1 = accuracy
                print("saving model ...")
                checkpoint_path = os.path.join(FLAGS.model_path, "translate.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
            
def unpack(batch):
    batchLen =[]
    right = []
    left = []
    aspects = []
    rightlen = []
    leftlen = []
    aspectlen = []
    polarity =[]
    for item in batch:
        polarity_array = [0,0,0]
        left.append(item.left)
        right.append(item.right)
        aspects.append(item.aspects)
        leftlen.append(item.leftLen)
        rightlen.append(item.rightLen)
        assert item.aspectlen != 0 , print("aspect len == 0")
        aspectlen.append(item.aspectlen)
        polarity_array[item.polarity] = 1
        polarity.append(polarity_array)
    '''
    print([len(i) for i in left])
    print([len(i) for i in right])
    print([len(i) for i in aspects])
    print(rightlen)
    print(leftlen)
    print(aspectlen)
    print([len(i) for i in polarity])
    '''
    left = np.asarray(left)
    right = np.asarray(right)
    aspects = np.asarray(aspects)
    rightlen = np.asarray(rightlen)
    leftlen = np.asarray(leftlen)
    aspectlen = np.asarray(aspectlen)
    polarity = np.asarray(polarity, dtype=np.float64)
    assert not np.any(np.isnan(left)), print("input Nan")
    assert not np.any(np.isnan(right)), print("input Nan")
    assert not np.any(np.isnan(aspects)), print("input Nan")
    assert not np.any(np.isnan(rightlen)), print("input Nan")
    assert not np.any(np.isnan(leftlen)), print("input Nan")
    assert not np.any(np.isnan(aspectlen)), print("input Nan")
    assert not np.any(np.isnan(polarity)), print("input Nan")
    '''
    print(left.shape)
    print(right.shape)
    print(aspects.shape)
    print(rightlen.shape)
    print(leftlen.shape)
    print(aspectlen.shape)
    print(polarity.shape)
    '''
    return {'left':left,
            'right':right,
            'aspects':aspects,
            'rightlen':rightlen,
            'leftlen':leftlen,
            'aspectlen':aspectlen,
            'polarity':polarity
            }
    
        
        
def get_batch(datasamples, batch_size):
    datalen = len(datasamples)
    batch_num = datalen // batch_size
    for i in range(0, batch_num):
        '''
        if i == batch_num:
            batch = datasamples[i*batch_size:]
        else:
        '''
        batch = datasamples[i*batch_size:(i+1)*batch_size]
        yield unpack(batch)
            
if __name__ == "__main__":
    tf.app.run(main)