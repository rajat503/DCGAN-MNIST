import tensorflow as tf
import numpy as np

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
x_discriminator_fake = tf.placeholder(tf.float32, shape=[None, 784])

d_x_image = tf.reshape(x_discriminator, [-1,28,28,1])
d_x_image_fake = tf.reshape(x_discriminator_fake, [-1,28,28,1])

d_W_conv1 = weight_variable([5,5,1,128])
d_b_conv1 = bias_variable([128])
d_conv2d_conv1 = tf.nn.conv2d(d_x_image, d_W_conv1, strides=[1, 2, 2, 1], padding='SAME') + d_b_conv1
d_h_conv1 =  tf.maximum(0.2*d_conv2d_conv1, d_conv2d_conv1)
d_conv2d_conv1_fake = tf.nn.conv2d(d_x_image_fake, d_W_conv1, strides=[1, 2, 2, 1], padding='SAME') + d_b_conv1
d_h_conv1_fake =  tf.maximum(0.2*d_conv2d_conv1_fake, d_conv2d_conv1_fake)


d_W_conv2 = weight_variable([5,5,128,256])
d_b_conv2 = bias_variable([256])
d_conv2d_conv2 = tf.nn.conv2d(d_h_conv1, d_W_conv2, strides=[1, 2, 2, 1], padding='SAME') + d_b_conv2
d_conv2d_conv2_mean, d_conv2d_conv2_var = tf.nn.moments(d_conv2d_conv2, axes=[0, 1, 2])
d_conv2d_conv2_bn = tf.nn.batch_normalization(d_conv2d_conv2, d_conv2d_conv2_mean, d_conv2d_conv2_var, None, None, variance_epsilon=0.00005)
d_h_conv2 =  tf.maximum(0.2*d_conv2d_conv2_bn, d_conv2d_conv2_bn)
d_conv2d_conv2_fake = tf.nn.conv2d(d_h_conv1_fake, d_W_conv2, strides=[1, 2, 2, 1], padding='SAME') + d_b_conv2
d_conv2d_conv2_mean_f, d_conv2d_conv2_var_f = tf.nn.moments(d_conv2d_conv2_fake, axes=[0, 1, 2])
d_conv2d_conv2_fake_bn = tf.nn.batch_normalization(d_conv2d_conv2_fake, d_conv2d_conv2_mean_f, d_conv2d_conv2_var_f, None, None, variance_epsilon=0.00005)
d_h_conv2_fake =  tf.maximum(0.2*d_conv2d_conv2_fake_bn, d_conv2d_conv2_fake_bn)


d_W_conv3 = weight_variable([5,5,256,512])
d_b_conv3 = bias_variable([512])
d_conv2d_conv3 = tf.nn.conv2d(d_h_conv2, d_W_conv3, strides=[1, 2, 2, 1], padding='SAME') + d_b_conv3
d_conv2d_conv3_mean, d_conv2d_conv3_var = tf.nn.moments(d_conv2d_conv3, axes=[0, 1, 2])
d_conv2d_conv3_bn = tf.nn.batch_normalization(d_conv2d_conv3, d_conv2d_conv3_mean, d_conv2d_conv3_var, None, None, variance_epsilon=0.00005)
d_h_conv3 =  tf.maximum(0.2*d_conv2d_conv3_bn, d_conv2d_conv3_bn)
d_conv2d_conv3_fake = tf.nn.conv2d(d_h_conv2_fake, d_W_conv3, strides=[1, 2, 2, 1], padding='SAME') + d_b_conv3
d_conv2d_conv3_mean_f, d_conv2d_conv3_var_f = tf.nn.moments(d_conv2d_conv3_fake, axes=[0, 1, 2])
d_conv2d_conv3_fake_bn = tf.nn.batch_normalization(d_conv2d_conv3_fake, d_conv2d_conv3_mean_f, d_conv2d_conv3_var_f, None, None, variance_epsilon=0.00005)
d_h_conv3_fake =  tf.maximum(0.2*d_conv2d_conv3_fake_bn, d_conv2d_conv3_fake_bn)

# d_W_conv4 = weight_variable([5,5,512,1024])
# d_b_conv4 = bias_variable([1024])
# d_conv2d_conv4 = tf.nn.conv2d(d_h_conv3, d_W_conv4, strides=[1, 2, 2, 1], padding='SAME') + d_b_conv4
# d_h_conv4 =  tf.maximum(0.2*d_conv2d_conv4, d_conv2d_conv4)

d_h_conv3_flat = tf.reshape(d_h_conv3, [-1, 4*4*512])
d_h_conv3_flat_fake = tf.reshape(d_h_conv3_fake, [-1, 4*4*512])

d_W_out = weight_variable([4*4*512, 1])
d_b_out = bias_variable([1])
d_out = tf.nn.sigmoid(tf.matmul(d_h_conv3_flat, d_W_out)+d_b_out)
d_out_fake = tf.nn.sigmoid(tf.matmul(d_h_conv3_flat_fake, d_W_out)+d_b_out)

d_loss = 0-tf.reduce_mean(tf.log(d_out) + tf.log(1-d_out_fake))
d_trainstep = tf.train.AdamOptimizer(learning_rate = 0.0002, beta1=0.5).minimize(d_loss)

#generator
x_generator = tf.placeholder(tf.float32, shape=[None, 100])
b = tf.placeholder(tf.int32)

g_W_input = weight_variable([100, 4*4*512])
g_b_input = bias_variable([4*4*512])
g_layer1 = tf.nn.relu(tf.matmul(x_generator, g_W_input)+g_b_input)

g_h_dconv1 = tf.reshape(g_layer1, [-1,4,4,512])

g_W_dconv2 = weight_variable([5, 5, 256, 512])
g_b_dconv2 = bias_variable([256])
g_dconv2 = tf.nn.conv2d_transpose(g_h_dconv1, g_W_dconv2, [b,7,7,256], strides=[1,2,2,1], padding='SAME') + g_b_dconv2
g_dconv2_mean, g_dconv2_var = tf.nn.moments(g_dconv2, axes=[0, 1, 2])
g_dconv2_bn = tf.nn.batch_normalization(g_dconv2, g_dconv2_mean, g_dconv2_var, None, None, variance_epsilon=0.00005)
g_h_dconv2 = tf.nn.relu(g_dconv2_bn)


g_W_dconv3 = weight_variable([5, 5, 128 ,256])
g_b_dconv3 = bias_variable([128])
g_dconv3 = tf.nn.conv2d_transpose(g_h_dconv2, g_W_dconv3, [b,14,14,128], strides=[1,2,2,1], padding='SAME') + g_b_dconv3
g_dconv3_mean, g_dconv3_var = tf.nn.moments(g_dconv3, axes=[0, 1, 2])
g_dconv3_bn = tf.nn.batch_normalization(g_dconv3, g_dconv3_mean, g_dconv3_var, None, None, variance_epsilon=0.00005)
g_h_dconv3 = tf.nn.relu(g_dconv3_bn)

g_W_dconv4 = weight_variable([5, 5, 1,128])
g_b_dconv4 = bias_variable([1])
g_h_dconv4 = tf.tanh(tf.nn.conv2d_transpose(g_h_dconv3, g_W_dconv4, [b,28,28,1], strides=[1,2,2,1], padding='SAME') + g_b_dconv4)

g_d_conv2d_conv1_fake = tf.nn.conv2d(g_h_dconv4*255, d_W_conv1, strides=[1, 2, 2, 1], padding='SAME') + d_b_conv1
g_d_h_conv1_fake =  tf.maximum(0.2*g_d_conv2d_conv1_fake, g_d_conv2d_conv1_fake)

g_d_conv2d_conv2_fake = tf.nn.conv2d(g_d_h_conv1_fake, d_W_conv2, strides=[1, 2, 2, 1], padding='SAME') + d_b_conv2
g_d_conv2d_conv2_mean_f, g_d_conv2d_conv2_var_f = tf.nn.moments(g_d_conv2d_conv2_fake, axes=[0, 1, 2])
g_d_conv2d_conv2_fake_bn = tf.nn.batch_normalization(g_d_conv2d_conv2_fake, g_d_conv2d_conv2_mean_f, g_d_conv2d_conv2_var_f, None, None, variance_epsilon=0.00005)
g_d_h_conv2_fake =  tf.maximum(0.2*g_d_conv2d_conv2_fake_bn, g_d_conv2d_conv2_fake_bn)

g_d_conv2d_conv3_fake = tf.nn.conv2d(g_d_h_conv2_fake, d_W_conv3, strides=[1, 2, 2, 1], padding='SAME') + d_b_conv3
g_d_conv2d_conv3_mean_f, g_d_conv2d_conv3_var_f = tf.nn.moments(g_d_conv2d_conv3_fake, axes=[0, 1, 2])
g_d_conv2d_conv3_fake_bn = tf.nn.batch_normalization(g_d_conv2d_conv3_fake, g_d_conv2d_conv3_mean_f, g_d_conv2d_conv3_var_f, None, None, variance_epsilon=0.00005)
g_d_h_conv3_fake =  tf.maximum(0.2*g_d_conv2d_conv3_fake_bn, g_d_conv2d_conv3_fake_bn)

g_d_h_conv3_flat_fake = tf.reshape(g_d_h_conv3_fake, [-1, 4*4*512])
g_d_out_fake = tf.nn.sigmoid(tf.matmul(g_d_h_conv3_flat_fake, d_W_out)+d_b_out)


g_loss = tf.reduce_mean(tf.log(1-g_d_out_fake))
g_trainstep = tf.train.AdamOptimizer(learning_rate = 0.0002, beta1=0.5).minimize(g_loss, var_list=[g_W_input, g_b_input, g_W_dconv2, g_b_dconv2, g_W_dconv3, g_b_dconv3, g_W_dconv4, g_b_dconv4])

sess.run(tf.initialize_all_variables())

for i in range(100):
    print "iteration", i
    batch_real = mnist.train.next_batch(128)[0]
    d_batch_fake = np.random.random_sample((128,100))

    g_z = sess.run(g_h_dconv4, feed_dict={x_generator: d_batch_fake, b: 128})
    _ , dl= sess.run([d_trainstep, d_loss], feed_dict = {x_discriminator_fake: (g_z*255).reshape([-1, 784]), x_discriminator: batch_real})

    print "discriminator loss",dl

    g_batch_fake = np.random.random_sample((128,100))
    x, gl = sess.run([g_trainstep, g_loss], feed_dict = {x_generator: g_batch_fake, b: 128})

    print "generator loss", gl


import cv2
img = cv2.imshow("a",(g_z[0]*255).reshape(28, 28))
k = cv2.waitKey(0) & 0xFF
if k == 27:
    cv2.destroyAllWindows()
