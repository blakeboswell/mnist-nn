import header
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets


def model(path, epoch_n, batch_size):

    mnist = read_data_sets(path, one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    
    y = tf.matmul(x, W) + b
    y_ = tf.placeholder(tf.float32, [None, 10])

    loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=y_, logits=y)
            )

    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    init = tf.global_variables_initializer()
    
    test_feed = {x: mnist.test.images, y_: mnist.test.labels}
    
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(epoch_n):
            xs, ys = mnist.train.next_batch(batch_size)
            train_feed = {x: xs, y_: ys}
            sess.run(train_step, feed_dict=train_feed)

            correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            
            metric = sess.run(accuracy, feed_dict=test_feed)
            header.train_message(epoch, metric)


if __name__ == '__main__':

    datapath = header.data_path( 'MNIST_data')
    epoch_n, batch_size = 1000, 100    
    
    model(datapath, epoch_n, batch_size)

