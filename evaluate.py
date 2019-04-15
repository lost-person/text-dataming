# coding = utf-8

import datetime
import logging
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.contrib import learn

import data_helpers
from text_cnn import TextCNN

out_dir = "./runs/NB/1554088006"
# 设置log
logging.basicConfig(level = logging.INFO, filename = os.path.join(out_dir, 'evaluate_htl_log.txt'), filemode = 'w')
logger = logging.getLogger(__name__)

# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("neg_data_path", "./data/htl_del_4000/neg_clean.txt", "negative data path.")
tf.flags.DEFINE_string("pos_data_path", "./data/htl_del_4000/pos_clean.txt", "positive data path.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", out_dir, "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.neg_data_path, FLAGS.pos_data_path)
# 获取真实标签
y_test = np.argmax(y_test, axis=1)

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = os.path.join(out_dir, 'checkpoints', 'model-350')
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])


correct_predictions = float(sum(all_predictions == y_test))
print("Total number of test examples: {}".format(len(y_test)))
logger.info("acc: {:g}".format(correct_predictions/float(len(y_test))))
logger.info("pre: {:g}".format(precision_score(y_test, all_predictions)))
logger.info("rec: {:g}".format(recall_score(y_test, all_predictions)))
logger.info("f1: {:g}".format(f1_score(y_test, all_predictions)))
df = pd.DataFrame({'real' : y_test, 'pre' : all_predictions})
df.to_csv(os.path.join(FLAGS.checkpoint_dir, 'htl_pre.csv'))