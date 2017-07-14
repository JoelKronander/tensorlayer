
"""
Introduction
----------------
This example demostrates how to perform semantic segmentation using
Fully Convolutional Neural Networks (FCN) in tensorlayer:
paper: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
original Caffe code: https://github.com/shelhamer/fcn.berkeleyvision.org

For this example we use the MIT Scene Parsing Challange 2016 dataset, labeling
pixels into one of 150 semantic categories, which include stuffs like sky,
road, grass, and discrete objects like person, car, bed.
"""
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import numpy as np
import scipy.misc
from tensorlayer import visualize

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

NUM_CLASSES = 150

#VGG16 model up to fully connected layers are used as a frontend module
def vgg16_model(net_in):
    with tf.name_scope('preprocess') as scope:
        """
        Notice that we include a preprocessing layer that takes the RGB image
        with pixels values in the range of 0-255 and subtracts the mean image
        values (calculated over the entire ImageNet training set).
        """
        mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
        net_in.outputs = net_in.outputs - mean
    """ conv1 """
    network = Conv2dLayer(net_in, act = tf.nn.relu, shape = [3, 3, 3, 64],      # 64 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv1_1')
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 64, 64],    # 64 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv1_2')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                padding='SAME', pool = tf.nn.max_pool, name ='pool1')
    """ conv2 """
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 64, 128],  # 128 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv2_1')
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 128, 128],  # 128 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv2_2')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                padding='SAME', pool = tf.nn.max_pool, name ='pool2')
    """ conv3 """
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 128, 256],  # 256 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv3_1')
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv3_2')
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 256, 256],  # 256 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv3_3')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                padding='SAME', pool = tf.nn.max_pool, name ='pool3')
    """ conv4 """
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 256, 512],  # 512 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv4_1')
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv4_2')
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv4_3')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                padding='SAME', pool = tf.nn.max_pool, name ='pool4')
    """ conv5 """
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv5_1')
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv5_2')
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [3, 3, 512, 512],  # 512 features for each 3x3 patch
                strides = [1, 1, 1, 1], padding='SAME', name ='conv5_3')
    network = PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                padding='SAME', pool = tf.nn.max_pool, name ='pool5')
    return network

#FCN-32 model specification
def fcn32_model(vgg16_model):
    #convolution layers on top of pool5 in vgg16
    network = Conv2dLayer(vgg16_model, act = tf.nn.relu, shape = [7, 7, 512, 4096],      # 64 features for each 3x3 patch
        strides = [1, 1, 1, 1], padding='SAME', name ='FCN_conv1')
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [1, 1, 4096, 4096],      # 64 features for each 3x3 patch
        strides = [1, 1, 1, 1], padding='SAME', name ='FCN_conv2')
    network = Conv2dLayer(network, act = tf.nn.relu, shape = [1, 1, 4096, NUM_CLASSES],      # 64 features for each 3x3 patch
        strides = [1, 1, 1, 1], padding='SAME', name ='FCN_conv3')

    #upsampling layer to get back to image size
    network = UpSampling2dLayer(network, size=[224, 224], is_scale=False)
    return network

def load_vgg16frontend_weights(sess, network):
    tl.files.maybe_download_and_extract('vgg16_weights.npz', 'data/',
        'https://www.cs.toronto.edu/~frossard/vgg16/', False)
    npz = np.load('data/vgg16_weights.npz')

    params = []
    for val in sorted( npz.items() ):
        #we are only intrested in the convolutional layers
        if(val[0][0:4]=='conv'):
            print("  Loading %s with shape %s" % (str(val[0]), str(val[1].shape)))
            params.append(val[1])

    tl.files.assign_params(sess, params, network)

def train_model(train_op, loss):
    print("Training FCN model with :")
    print('   learning_rate: %f' % FLAGS.learning_rate)
    print('   batch_size: %d' % FLAGS.batch_size)

    n_epoch = 10
    print_freq = 1
    for epoch in range(n_epoch):
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate.minibatches(
                                    x_train, y_train, FLAGS.batch_size, shuffle=True):
            _, err = sess.run((train_op, loss), feed_dict={x: X_train_a, y: y_train_a})
            print("loss %f" % err)

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
            test_loss, test_acc, n_batch = 0, 0, 0
            for X_test_a, y_test_a in tl.iterate.minibatches(
                                        x_test, y_test, batch_size, shuffle=False):
                err = sess.run(loss, feed_dict={x: X_train_a, y: y_train_a})
                test_loss += err; n_batch += 1
            print("   test loss: %f" % (test_loss/ n_batch))

def evaluate_model():
    pass

if __name__ == "__main__":

    #Load a random subset of the MITSceneParsingChallange 2016 dataset
    x_train, y_train, x_test, y_test = tl.files.load_MITsceneparsing_dataset(load_size=1,randomize=True)

    x = tf.placeholder(tf.float32, [None, 224, 224, 3])
    y = tf.placeholder(tf.int32, [None, 224, 224, 1])

    net_in = InputLayer(x, name='input')
    vgg16_model = vgg16_model(net_in);
    model = fcn32_model(vgg16_model)
    argmax_pred = tf.argmax(model.outputs, dimension=3, name="prediction1")
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=model.outputs,labels=tf.squeeze(y, squeeze_dims=[3]), name="entropy")))

    train_params = model.all_params
    train_op = tf.train.AdamOptimizer(FLAGS.learning_rate, beta1=0.9, beta2=0.999,
        epsilon=1e-08, use_locking=False).minimize(loss, var_list=train_params)
    #pred = tf.expand_dims(pred, dim=3)

    sess = tf.InteractiveSession()
    tl.layers.initialize_global_variables(sess)
    model.print_layers()
    load_vgg16frontend_weights(sess, vgg16_model)
    #model.print_params()

    img1 = scipy.misc.imread('data/laska.png', mode='RGB') # test data in github
    img1 = scipy.misc.imresize(img1, (224, 224))

    start_time = time.time()
    prob = sess.run(loss, feed_dict={x: x_train[0:1], y: y_train[0:1]})
    print("  End time : %.5ss" % (time.time() - start_time))
    print prob

    train_model(train_op, loss)
    #evaluate_model()
