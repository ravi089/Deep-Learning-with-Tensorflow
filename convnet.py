# Convolutional Neural Network
# <Hand Written Digit Classification>
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Hyper parameters.
learning_rate = 0.001
dropout = 0.8
training_epochs = 200
n_input = 784
n_classes = 10
batch_size = 128
display_step = 10

X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# Weights => [filter_size, filter_size, n_channels, n_filters]
# Biases => [n_filters]
net_param = {
    'layer_1': {
        'weight': tf.Variable(tf.random_normal([5, 5, 1, 32])),
        'bias': tf.Variable(tf.random_normal([32]))
    },
    'layer_2': {
        'weight': tf.Variable(tf.random_normal([5, 5, 32, 64])),
        'bias': tf.Variable(tf.random_normal([64]))
    },
    'layer_3': {
        'weight': tf.Variable(tf.random_normal([7*7*64, 1024])),
        'bias': tf.Variable(tf.random_normal([1024]))
    },
    'layer_out': {
        'weight': tf.Variable(tf.random_normal([1024, n_classes])),
        'bias': tf.Variable(tf.random_normal([n_classes]))
    }   
}

# Convolutional Layer.
def conv_layer(x, W, b, strides = 1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

# Pooling Layer.
def pool_layer(x, k = 2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')

# Create Conv Net.
def conv_net(x, net_param, dropout):
    # Transform 1-D vector of 784 features into input
    # format for conv net [batch size, height, width, channel].
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolutional Layer.
    conv1 = conv_layer(x, net_param['layer_1']['weight'], net_param['layer_1']['bias'])
    conv1 = pool_layer(conv1)

    # Convolutional Layer.
    conv2 = conv_layer(conv1, net_param['layer_2']['weight'], net_param['layer_2']['bias'])
    conv2 = pool_layer(conv2)

    # Fully Connected Layer.
    fc1 = tf.reshape(conv2, [-1, net_param['layer_3']['weight'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, net_param['layer_3']['weight']), net_param['layer_3']['bias'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, dropout)

    # Class prediction.
    out = tf.add(tf.matmul(fc1, net_param['layer_out']['weight']), net_param['layer_out']['bias'])
    return out

# Construct model.
logits = conv_net(X, net_param, keep_prob)
prediction = tf.nn.softmax(logits)

# Loss and Optimizer.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                            logits = logits, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Evaluation.
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Training.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1, training_epochs + 1):
        train_x, train_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict = {X: train_x,
                                         Y: train_y,
                                         keep_prob: dropout})
        if step % display_step == 0 or step == 1:
            cost, acc = sess.run([loss, accuracy], feed_dict = {X: train_x,
                                                                Y: train_y,
                                                                keep_prob: 1.0})
            print ('Step ' + str(step) + ', Minibatch Loss = ' + \
                   '{:.4f}'.format(cost) + ', Training Accuracy = ' + \
                   '{:.4f}'.format(acc))
    print ('Training Done!')
    print ('Testing Accuracy:', \
           sess.run(accuracy, feed_dict = {X: mnist.test.images[:300],
                                           Y: mnist.test.labels[:300],
                                           keep_prob: 1.0}))
