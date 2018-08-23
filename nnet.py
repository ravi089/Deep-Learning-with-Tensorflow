# 3-Layer Neural Network <Hand Written Digit Classification>
import tensorflow as tf

# Import MNIST data.
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

# Hyper parameters.
learning_rate = 0.001
training_epochs = 1000
n_hidden_1 = 380    
n_hidden_2 = 460  
n_input = 784   
n_classes = 10  
batch_size = 128
display_step = 100

X = tf.placeholder('float', [None, n_input])
Y = tf.placeholder('float', [None, n_classes])

# Weights and Biases.
net_param = {
    'layer_1': {
        'weight': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'bias': tf.Variable(tf.random_normal([n_hidden_1]))
    },
    'layer_2': {
        'weight': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'bias': tf.Variable(tf.random_normal([n_hidden_2]))
    },
    'layer_out': {
        'weight': tf.Variable(tf.random_normal([n_hidden_2, n_classes])),
        'bias': tf.Variable(tf.random_normal([n_classes]))
    }
}

# Create Neural network.
def neural_network(input_layer):
    layer_1 = tf.add(tf.matmul(input_layer, net_param['layer_1']['weight']), net_param['layer_1']['bias'])
    layer_2 = tf.add(tf.matmul(layer_1, net_param['layer_2']['weight']), net_param['layer_2']['bias'])
    out_layer = tf.add(tf.matmul(layer_2, net_param['layer_out']['weight']), net_param['layer_out']['bias'])
    return out_layer

logits = neural_network(X)
prediction = tf.nn.softmax(logits)

# Loss(Cross Entropy Loss) and Optimizer.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                            logits = logits, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

# Evaluation.
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Training.
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1, training_epochs + 1):
        train_x, train_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict = {X: train_x, Y: train_y})
        if step % display_step == 0 or step == 1:
            cost, acc = sess.run([loss, accuracy], feed_dict = {X: train_x,
                                                                Y: train_y})
            print ('Step ' + str(step) + ', Minibatch Loss = ' + \
                   '{:.4f}'.format(cost) + ', Training Accuracy = ' + \
                   '{:.4f}'.format(acc))
    print ('Training Done!!!')
    print ('Testing Accuracy:', \
           sess.run(accuracy, feed_dict = {X: mnist.test.images,
                                           Y: mnist.test.labels}))

