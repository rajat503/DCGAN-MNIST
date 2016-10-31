'''
TODO: batch-norm
      discriminator loss function
'''

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.02)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.02)
  return tf.Variable(initial)

#discriminator with leaky relu and stride 2
x_discriminator = tf.placeholder(tf.float32, shape=[None, 784])

d_x_image = tf.reshape(x_discriminator, [-1,28,28,1])

d_W_conv1 = weight_variable([5,5,1,128])
d_b_conv1 = bias_variable([128])
d_conv2d_conv1 = tf.nn.conv2d(d_x_image, d_W_conv1, strides=[1, 2, 2, 1], padding='SAME') + d_b_conv1
d_h_conv1 =  tf.maximum(0.2*d_conv2d_conv1, d_conv2d_conv1)

d_W_conv2 = weight_variable([5,5,128,256])
d_b_conv2 = bias_variable([256])
d_conv2d_conv2 = tf.nn.conv2d(d_h_conv1, d_W_conv2, strides=[1, 2, 2, 1], padding='SAME') + d_b_conv2
d_h_conv2 =  tf.maximum(0.2*d_conv2d_conv2, d_conv2d_conv2)

d_W_conv3 = weight_variable([5,5,256,512])
d_b_conv3 = bias_variable([512])
d_conv2d_conv3 = tf.nn.conv2d(d_h_conv2, d_W_conv3, strides=[1, 2, 2, 1], padding='SAME') + d_b_conv3
d_h_conv3 =  tf.maximum(0.2*d_conv2d_conv3, d_conv2d_conv3)

# d_W_conv4 = weight_variable([5,5,512,1024])
# d_b_conv4 = bias_variable([1024])
# d_conv2d_conv4 = tf.nn.conv2d(d_h_conv3, d_W_conv4, strides=[1, 2, 2, 1], padding='SAME') + d_b_conv4
# d_h_conv4 =  tf.maximum(0.2*d_conv2d_conv4, d_conv2d_conv4)

d_h_conv3_flat = tf.reshape(d_h_conv3, [-1, 4*4*512])

d_W_out = weight_variable([4*4*512, 1])
d_b_out = bias_variable([1])
d_out = tf.nn.sigmoid(tf.matmul(d_h_conv3_flat, d_W_out)+d_b_out)



#generator
# x_generator = tf.placeholder(tf.float32, shape=[None, 100])
#







sess.run(tf.initialize_all_variables())

#making the calls

#sample from distribution

#sample from real dataset
