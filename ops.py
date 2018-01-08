import tensorflow as tf
import tensorflow.contrib.slim as slim

def batch_norm(x, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True, scope=name)

def instance_norm(input,name="instance_norm"):
    with tf.variable_scope(name):
        depth = input.get_shape()[3]
        scale = tf.get_variable("scale",shape=[depth],
                                initializer= tf.random_normal_initializer(mean=1.0,stddev=0.02,dtype=tf.float32))
        offset = tf.get_variable("offset", shape=[depth],initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input,axes=[1,2], keep_dims= True)
        epsilon = 1e-5
        inv= tf.rsqrt(variance+epsilon)
        normalized = (input - mean)*inv
        return scale*normalized + offset

def lin_instance_norm(input,name="lin_instance_norm"):
    with tf.variable_scope(name):
        scale = tf.get_variable("scale",shape=[1],
                                initializer= tf.random_normal_initializer(mean=1.0,stddev=0.02,dtype=tf.float32))
        offset = tf.get_variable("offset", shape=[1], initializer=tf.constant_initializer(0.0))
        mean, variance = tf.nn.moments(input,axes=[1], keep_dims= True)
        epsilon = 1e-5
        inv= tf.rsqrt(variance+epsilon)
        normalized = (input - mean)*inv
        return scale*normalized + offset

def conv2d(input, out_dim, ks=4, s=2, padding = 'SAME',stddev = 0.02, name='conv2d'):
    with tf.variable_scope(name):
        return slim.conv2d(input, out_dim,ks,s,padding= padding,activation_fn=None,
                                 weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                 biases_initializer=None)

def deconv2d(input, out_dim, ks=4, s=2, stddev=0.02, name='deconv2d'):
    with tf.variable_scope(name):
        return slim.conv2d_transpose(input,out_dim,ks, s, padding='SAME',activation_fn=None,
                                     weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
                                     biases_initializer=None)
def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(leak*x, x,name= name)

def linear(input, out_size, stddev=0.02, bias_initializer=0.0, name ="lin", with_w = False):
    with tf.variable_scope(name):
        weights = tf.get_variable('weights', shape=[input.get_shape()[-1], out_size], dtype=tf.float32,
                                  initializer=tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable(name='bias',shape=[out_size],dtype=tf.float32,
                               initializer=tf.constant_initializer(bias_initializer))
        if with_w:
            return tf.matmul(input, weights) + bias, weights, bias
        else:
            return tf.matmul(input, weights) + bias

def conv_concat(x,y):
    x_size =x.get_shape()
    y_size =y.get_shape()
    return tf.concat([x, y * tf.ones([x_size[0],x_size[1],x_size[2],y_size[3]])], axis = 3)

