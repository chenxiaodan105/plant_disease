
# coding: utf-8

# In[ ]:


import tensorflow as tf
LEARNINGRATE = 1e-3

def weight_variable(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)


def bias_variable(shape, bais=0.1):
    initial = tf.constant(bais, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, [1, 1, 1, 1], 'SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')


def max_pool_3x3(x):
    return tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')


def avg_pool_3x3(x):
    return tf.nn.avg_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')


def inference(features, one_hot_labels):
    keep_prob = tf.placeholder('float')
    # network structure
    # conv1
    W_conv1 = weight_variable([5, 5, 3, 64], stddev=1e-4)
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d(features, W_conv1) + b_conv1)
    h_pool1 = max_pool_3x3(h_conv1)
    # norm1
    norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
    
    # fc1
    W_fc1 = weight_variable([64 * 64 * 64, 61])
    b_fc1 = bias_variable([61])
    h_pool3_flat = tf.reshape(norm1, [-1, 64*64*64])
    #h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
    
    y_conv = tf.matmul(h_pool3_flat, W_fc1) + b_fc1
    
    # calculate loss
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=y_conv))
    train_step = tf.train.AdamOptimizer(LEARNINGRATE).minimize(cross_entropy)
    
    return train_step, cross_entropy, y_conv, keep_prob

