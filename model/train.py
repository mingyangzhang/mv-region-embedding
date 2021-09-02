from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from utils import *
from models import *
from tasks import predict_crime, lu_classify

# Set random seed
seed = 1
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 2000, 'Number of epochs to train.')
flags.DEFINE_boolean('load', False, 'Whether load session.')
flags.DEFINE_integer('output_dim', 96, 'Number of hidden units in gcn.')

flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_float('weight_decay', 1e-3, 'Weight for L2 loss on embedding matrix.')

# Load data
data = load_data()

mob_adj = data["mob_adj"]

s_adj_sp = data["s_adj_sp"]
t_adj_sp = data["t_adj_sp"]

poi_adj = data["poi_adj"]
poi_adj_sp = data["poi_adj_sp"]

chk_adj = data["chk_adj"]
chk_adj_sp = data["chk_adj_sp"]

feature = data["feature"]
time_step, item_num, _ = mob_adj.shape
mob_shape = (time_step, item_num, item_num)
adj_shape = (item_num, item_num)
bias_mat_shape = (1, item_num, item_num)

mask_shape = (item_num, item_num, 2)

# Some preprocessing
feature = preprocess_features(feature)

data["s_bias"] = adj_to_bias(s_adj_sp, [item_num], 1)
data["t_bias"] = adj_to_bias(t_adj_sp, [item_num], 1)
data["poi_bias"] = adj_to_bias(poi_adj_sp, [item_num], 1)
data["chk_bias"] = adj_to_bias(chk_adj_sp, [item_num], 1)

# Define placeholders
placeholders = {
    'mob_adj': tf.placeholder(tf.float32, shape=mob_shape),
    'poi_adj': tf.placeholder(tf.float32, shape=adj_shape),
    'chk_adj': tf.placeholder(tf.float32, shape=adj_shape),

    's_bias': tf.placeholder(tf.float32, shape=bias_mat_shape),
    't_bias': tf.placeholder(tf.float32, shape=bias_mat_shape),
    'poi_bias': tf.placeholder(tf.float32, shape=bias_mat_shape),
    'chk_bias': tf.placeholder(tf.float32, shape=bias_mat_shape),

    'feature': tf.placeholder(tf.float32, shape=feature.shape),
}

# Create model
model = MVURE(placeholders, input_dim=feature.shape[2], output_dim=FLAGS.output_dim, logging=True)
# Initialize session
sess = tf.Session()

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('logs/train', sess.graph)

# Init variables
sess.run(tf.global_variables_initializer())

if FLAGS.load:
    model.load(sess)

# Train model
iteration = 0
for epoch in range(FLAGS.epochs):
    # Construct feed dictionary
    feed_dict = construct_feed_dict(data, placeholders)
    # Training step
    outs = sess.run([model.opt_op, model.loss, model.out, merged], feed_dict=feed_dict)
    train_writer.add_summary(outs[-1], iteration)
    iteration += 1
    output = np.squeeze(outs[-2])
    # test on two tasks

    if (epoch+1) % 500 == 0:
        print("Train loss {:.3f} at epoch {}.".format(outs[1], epoch))

predict_crime(output)
lu_classify(output)

model.save(sess)
print("Optimization Finished!")
