from __future__ import division
import tensorflow as tf
from ops import *
from utils import *

def generator(z,y,option,reuse = False, name="g"):

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False
        im_size = 8
        yl = tf.reshape(y, [option.batch_size,1,1,10])
        z = tf.concat([z,y],1)

        h0 = tf.nn.relu(batch_norm(linear(z, 1024,name = "g_h0_lin" ),name="d_bn0"))
        h0 = tf.concat([h0,y],1)

        h1 = tf.nn.relu(batch_norm(
            linear(h0,option.gf_dim*2*im_size*im_size,name = "g_h1_lin"),name="d_bn1"))
        h1= tf.reshape(h1, [option.batch_size,im_size,im_size,option.gf_dim*2])
        h1 = conv_concat(h1,yl)

        h2 = tf.nn.relu(batch_norm(deconv2d(h1,option.gf_dim*2, ks=5,s=2,name="g_h2_deconv"),"d_bn2"))
        h2 = conv_concat(h2,yl)

        h3 = tf.nn.tanh(deconv2d(h2,option.output_c_dim,5,2, name="g_h3_deconv"))

        return h3
def buffer(gimage,option,reuse = False, name="buffer"):

    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False
        # print('gmimage shape', gimage.get_shape())
        h1 = tf.nn.relu(batch_norm(conv2d(gimage,option.gf_dim/16, ks=4,s=1,name="buffer_h1_conv"),"buffer_bn1"))
        # print('h1 shape',h1.get_shape())
        return tf.nn.tanh(conv2d(h1,option.output_c_dim,4,1, name="buffer_h2_conv"))


def discriminator(image, y,option, reuse=False, name="dis"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        yl = tf.reshape(y, [option.batch_size,1,1,10])


        x = conv_concat(image, yl)

        h0 = lrelu(conv2d(x,1+10,5,2,name="d_h0_conv"))
        h0 = conv_concat(h0,yl)

        h1 =lrelu(batch_norm(conv2d(h0,64+10,5,2,name="d_h1_conv"),"d_bn1"))
        h1 = tf.reshape(h1,[option.batch_size,-1])
        h1 =tf.concat([h1,y], 1)

        h2 = lrelu(batch_norm(linear(h1, 1024, name="d_h2_linear"),'d_bn2'))
        h2= tf.concat([h2,y],1)

        h3 = linear(h2,1,name="d_h3_linear")

        return h3

def patch_d(image, options, reuse=False, name="patch_d"):

    with tf.variable_scope(name):
        # image is 32 x 32 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        h0 = lrelu(conv2d(image, options.df_dim, ks=3,s=1, name='d_h0_conv'))
        # h0 is (16 x 16 x self.df_dim)
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2,ks=1,s=1, name='d_h1_conv'), 'd_bn1'))
        # h1 is (8 x 8 x self.df_dim*2)
        h2 = conv2d(h1, 1, ks=1, s=1, name='d_h2_pred')
        return h2


def abs(x,y):
    return tf.reduce_mean(tf.abs(x,y))

def mae(x,y):
    return tf.reduce_mean((x-y)**2)

def sce(x,y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=x,labels=y))
# canny--------------------------------------------------------------------------------------------------

def conv(input, out_dim=1, ks=1, s=1, padding = 'SAME',value = [0.114, 0.587, 0.299], name='constant_conv'):
    with tf.variable_scope(name):
        return slim.conv2d(input, out_dim,ks,s,padding= padding,activation_fn=None,
                                 weights_initializer=tf.constant_initializer(value=value),
                                 biases_initializer=tf.constant_initializer(0))

def rgb2gray(image): # Gray processing
    c = image.get_shape()[3]
    if c==3:
        gray = conv(image,out_dim=1,ks=1,s=1,value = [0.0721, 0.7154, 0.2125], name='gray')
        return gray
    else:
        return image

def smooth(gray):
    value = [1.79106361e-08, 5.93118809e-07, 7.22566631e-06,
             3.23831897e-05, 5.33908537e-05, 3.23831897e-05,
             7.22566631e-06, 5.93118809e-07, 1.79106361e-08,
             5.93118809e-07, 1.96413974e-05, 2.39281205e-04,
             1.07238396e-03, 1.76806225e-03, 1.07238396e-03,
             2.39281205e-04, 1.96413974e-05, 5.93118809e-07,
             7.22566631e-06, 2.39281205e-04, 2.91504184e-03,
             1.30643112e-02, 2.15394077e-02, 1.30643112e-02,
             2.91504184e-03, 2.39281205e-04, 7.22566631e-06,
             3.23831897e-05, 1.07238396e-03, 1.30643112e-02,
             5.85501805e-02, 9.65329280e-02, 5.85501805e-02,
             1.30643112e-02, 1.07238396e-03, 3.23831897e-05,
             5.33908537e-05, 1.76806225e-03, 2.15394077e-02,
             9.65329280e-02, 1.59155892e-01, 9.65329280e-02,
             2.15394077e-02, 1.76806225e-03, 5.33908537e-05,
             3.23831897e-05, 1.07238396e-03, 1.30643112e-02,
             5.85501805e-02, 9.65329280e-02, 5.85501805e-02,
             1.30643112e-02, 1.07238396e-03, 3.23831897e-05,
             7.22566631e-06, 2.39281205e-04, 2.91504184e-03,
             1.30643112e-02, 2.15394077e-02, 1.30643112e-02,
             2.91504184e-03, 2.39281205e-04, 7.22566631e-06,
             5.93118809e-07, 1.96413974e-05, 2.39281205e-04,
             1.07238396e-03, 1.76806225e-03, 1.07238396e-03,
             2.39281205e-04, 1.96413974e-05, 5.93118809e-07,
             1.79106361e-08, 5.93118809e-07, 7.22566631e-06,
             3.23831897e-05, 5.33908537e-05, 3.23831897e-05,
             7.22566631e-06, 5.93118809e-07, 1.79106361e-08]
    smooth = conv(gray,ks=9, s =1,value= value,padding='SAME', name= 'smooth')
    bleed_over = conv(tf.ones_like(gray), ks=9, s=1, value=value, name='bleed_over')
    smooth = smooth/(bleed_over+tf.ones_like(gray)*np.finfo(float).eps)
    return smooth

def sobel_xy(smooth):
    smooth = tf.pad(smooth, [[0, 0], [1, 1], [1, 1], [0, 0]], "SYMMETRIC")
    x = [-1, 0, +1,
         -2, 0, +2,
         -1, 0, +1]
    y = [+1, +2, +1,
          0,  0,  0,
         -1, -2, -1]
    Gy = conv(smooth, ks=3, s=1, value=y, padding='VALID', name='Gy')
    Gx = conv(smooth, ks=3, s=1, value=x, padding='VALID', name='Gx')

    return Gy, Gx

def magnitude(Gy,Gx,low,high):
    G = tf.sqrt(tf.add(tf.square(Gy),tf.square(Gx)))
    G= tf.nn.tanh(5.9 * (G - (high-low)/2)/(high/low))
    # G= tf.nn.tanh(high*G -low)
    return G

def sobel(image,low =0.1*2, high=0.2*2, reuse=False, name="canny"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False
        gray = rgb2gray(image)
        smoothed = smooth(gray)
        Gy, Gx = sobel_xy(smoothed)
        G = magnitude(Gy, Gx,low, high)
        return G

def discriminator_svmn(image, options, reuse=False, name="discriminator"):

    with tf.variable_scope(name):
        # image is 32 x 32 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse == False

        h0 = lrelu(conv2d(image, options.df_dim, ks=4,s=2, name='d_h0_conv'))
        # h0 is (16 x 16 x self.df_dim)
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2,ks=4,s=2, name='d_h1_conv'), 'd_bn1'))
        # h1 is (8 x 8 x self.df_dim*2)
        # h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4,ks=5,s=2, name='d_h2_conv'), 'd_bn2'))
        # # h2 is (4x 4 x self.df_dim*4)
        # h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, ks=5,s=2, name='d_h3_conv'), 'd_bn3'))
        # # h3 is (2 x 2 x self.df_dim*8)
        h2 = conv2d(h1, 1, ks=4, s=2, name='d_h2_pred')
        # h4 is (4 x 4 x 1)
        return h2