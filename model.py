from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
from collections import namedtuple
from module import *
from utils import *

class amgan(object):
    def __init__(self, sess, args):
        self.sess= sess
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.gf_dim = args.ngf
        self.df_dim = args.ndf
        self.z_dim = args.z_dim
        self.inputA_c_dim = args.inputA_nc
        self.inputB_c_dim = args.inputB_nc
        self.image_size = args.fine_size
        self.g = generator
        self.d = discriminator
        self.sobel = sobel
        self.patch_d = patch_d
        self.buffer = buffer
        self.lsgan = mae
        OPTIONS = namedtuple('OPTIONS', 'batch_size image_size \
                              gf_dim df_dim output_c_dim')
        self.options = OPTIONS._make((args.batch_size, args.fine_size,
                                    args.ngf, args.ndf, args.inputB_nc))

        self._build_model()
        self.saver = tf.train.Saver()

    def _build_model(self):

        self.real_A = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size, self.image_size,
                                         self.inputA_c_dim],
                                        name="real_A")
        self.real_B = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size, self.image_size,
                                         self.inputB_c_dim],
                                        name="real_B")

        self.y = tf.placeholder(tf.float32,[None,10],name="label_y")
        self.z = tf.placeholder(tf.float32,[None, self.z_dim], name="noise_z")

        self.fake_A = self.g(self.z,self.y,self.options, reuse = False, name="g_a" )
        self.fake_A_buffer = self.buffer(self.fake_A, self.options,reuse=False,name="buffer")
        self.fake_A_shape = self.sobel(self.fake_A_buffer, low =0.2, high=0.4,name="a")
        self.dfake_A_shape = self.d(self.fake_A_shape,self.y,self.options,  reuse= False, name="d_a")
        self.real_A_shape = self.sobel(self.real_A, low =0.2, high=0.4,name="as")
        self.dreal_A_shape = self.d(self.real_A_shape, self.y,self.options, reuse = True, name="d_a")

        self.g_loss_a = self.lsgan(self.dfake_A_shape, tf.ones_like(self.dfake_A_shape))
        self.d_loss_real_a = self.lsgan(self.dreal_A_shape,tf.ones_like(self.dreal_A_shape))
        self.d_loss_fake_a = self.lsgan(self.dfake_A_shape,tf.zeros_like(self.dfake_A_shape))
        self.d_loss_a= (self.d_loss_real_a + self.d_loss_fake_a)

        self.dfake_B = self.patch_d(self.fake_A, self.options, reuse=False, name="patch_d")
        self.dreal_B = self.patch_d(self.real_B, self.options, reuse=True, name="patch_d")

        self.g_loss_b = self.lsgan(self.dfake_B, tf.ones_like(self.dfake_B))
        self.d_loss_real_b = self.lsgan(self.dreal_B, tf.ones_like(self.dreal_B))
        self.d_loss_fake_b = self.lsgan(self.dfake_B, tf.zeros_like(self.dfake_B))
        self.d_loss_b = (self.d_loss_real_b + self.d_loss_fake_b)

        self.g_loss = self.g_loss_a+0.15*self.g_loss_b
        self.d_loss = self.d_loss_a+0.01*self.d_loss_b

        self.g_a_sum = tf.summary.scalar("g_loss_a", self.g_loss_a)
        self.d_a_sum = tf.summary.scalar("da_loss", self.d_loss_a)
        self.d_loss_real_a_sum = tf.summary.scalar("da_loss_real", self.d_loss_real_a)
        self.d_loss_fake_a_sum = tf.summary.scalar("da_loss_fake", self.d_loss_fake_a)
        self.da_sum = tf.summary.merge(
            [self.d_a_sum,self.d_loss_real_a_sum,self.d_loss_fake_a_sum]
        )
        self.g_b_sum = tf.summary.scalar("g_loss_b", self.g_loss_b)
        self.d_b_sum = tf.summary.scalar("db_loss", self.d_loss_b)
        self.d_loss_real_b_sum = tf.summary.scalar("db_loss_real", self.d_loss_real_b)
        self.d_loss_fake_b_sum = tf.summary.scalar("db_loss_fake", self.d_loss_fake_b)
        self.db_sum = tf.summary.merge(
            [self.d_b_sum, self.d_loss_real_b_sum, self.d_loss_fake_b_sum]
        )
        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        self.g_sum = tf.summary.merge(
            [self.g_loss_sum, self.g_a_sum, self.g_b_sum]
        )
        self.d_sum = tf.summary.merge(
            [self.d_loss_sum, self.d_a_sum,self.d_loss_real_a_sum,self.d_loss_fake_a_sum,
             self.d_b_sum, self.d_loss_real_b_sum, self.d_loss_fake_b_sum]
        )


        t_vars = tf.trainable_variables()
        self.da_vars = [var for var in t_vars if "d_a" in var.name]
        self.buffer_vars = [var for var in t_vars if "buffer" in var.name]
        self.ga_vars = [var for var in t_vars if "g_a" in var.name]
        self.db_vars = [var for var in t_vars if "patch_d" in var.name]
        for var in t_vars: print("VAR",var.name)

    def train(self, args):

        self.d_optim = tf.train.AdamOptimizer(args.lr, beta1= args.beta1).\
            minimize(self.d_loss, var_list= [self.da_vars,self.db_vars])
        self.g_optim = tf.train.AdamOptimizer(args.lr, beta1 = args.beta1).\
            minimize(self.g_loss, var_list= [self.ga_vars,self.buffer_vars])
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        self.writer = tf.summary.FileWriter(args.log_dir, self.sess.graph)

        counter = 1
        start_time = time.time()

        if self.load(args.checkpoint_dir):
            print("[*] Load SUCCESS")
        else:
            print("[!] Load Failed...")

        for epoch in range(args.epoch):
            dataA = loadimage('/home/guangyuan/DCQ/svhn.mat')['X']
            dataB = loadimage('/home/guangyuan/DCGAN/mnist32.mat')['X']
            np.random.shuffle(dataB)
            batch_idxs = min(len(dataA),len(dataB), args.train_size) // args.batch_size
            for idx in range(batch_idxs):
                batch_files = dataA[idx* args.batch_size: (idx+1)*args.batch_size]
                batch_images = load_data(batch_files).astype(np.float32)
                batch_labels = load_label(batch_files).astype(np.float32)
                batch_files_b = dataB[idx* args.batch_size: (idx+1)*args.batch_size]
                batch_images_b = load_data(batch_files_b).astype(np.float32)
                batch_z = np.random.uniform(-1,1, size=(args.batch_size, 100)).astype(np.float32)


                _, summary_str = self.sess.run([self.d_optim, self.d_sum],
                                               feed_dict={self.real_A: batch_images,
                                                          self.real_B: batch_images_b,
                                                          self.y: batch_labels, self.z: batch_z})
                self.writer.add_summary(summary_str, counter)

                _, summary_str = self.sess.run([self.g_optim, self.g_sum],
                                               feed_dict={self.y: batch_labels, self.z: batch_z}
                                               )
                self.writer.add_summary(summary_str, counter)

                _, summary_str = self.sess.run([self.g_optim, self.g_sum],
                                               feed_dict={self.y: batch_labels, self.z: batch_z}
                                               )
                self.writer.add_summary(summary_str, counter)

                counter += 1
                print(("Epoch: [%2d] [%4d/%4d] time: %4.4f"\
                      %(epoch, idx, batch_idxs, time.time()-start_time)))
                if np.mod(counter,100)==0:
                    self.sample_model(args.sample_dir, epoch, idx)
                if np.mod(counter,1000)==2:
                    self.save(args.checkpoint_dir, counter)

    def save(self, checkpoint_dir, step):
        model_name ="amgan.model"
        model_dir = "%s_%s" %(self.dataset_name, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir,model_dir)
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir,model_name),
                        global_step=step)
    def load(self, checkpoint_dir):
        print("[*] Reading checkpoint...")
        model_dir = "%s_%s" %(self.dataset_name, self.image_size)
        checkpoint_dir = os.path.join(checkpoint_dir,model_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def sample_model(self,sample_dir, epoch, idx):
        dataA = loadimage('/home/guangyuan/DCQ/svhn.mat')['X']
        dataB = loadimage('/home/guangyuan/DCGAN/mnist32.mat')['X']
        batch_files = dataA[0 * 64: (0 + 1) * 64]
        batch_images = load_data(batch_files).astype(np.float32)
        batch_labels = load_label(batch_files).astype(np.float32)
        batch_z = np.random.uniform(-1, 1, size=(64, 100)).astype(np.float32)
        [real_shape, img, buffer, fake_shape] = self.sess.run([self.real_A_shape,self.fake_A,self.fake_A_buffer,self.fake_A_shape],
                               feed_dict = {self.real_A:batch_images,self.z:batch_z,self.y: batch_labels})
        # print(real_shape.shape,fake_shape.shape)

        save_images(img,[8,8],
                    './{}/img_{:2d}_{:4d}'.format(sample_dir,epoch,idx))
        save_images(buffer, [8, 8],
                    './{}/buffer_{:2d}_{:4d}'.format(sample_dir, epoch, idx))
        save_images(fake_shape, [8, 8],
                    './{}/fs_{:2d}_{:4d}'.format(sample_dir, epoch, idx))

        save_images(real_shape, [8, 8],
                    './{}/rs_{:2d}_{:4d}'.format(sample_dir, epoch, idx))

    def test(self, args):
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)
        if self.load(args.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        yp = np.zeros(10 * self.batch_size).reshape(self.batch_size, 10)
        for i in range(self.batch_size):
            j = i % 10
            yp[i, j] = 1

        for k in range(100):

            z1 = (np.random.uniform(-1, 1, size=(1, 100)).astype(np.float32)) * np.ones(
                [int(np.sqrt(self.batch_size)), 100])
            z2 = (np.random.uniform(-1, 1, size=(1, 100)).astype(np.float32)) * np.ones(
                [int(np.sqrt(self.batch_size)), 100])
            batch_z = z1
            for idx, ratio in enumerate(np.linspace(0, 1, 8)):
                z = np.stack([slerp(ratio, r1, r2) for r1, r2 in zip(z1, z2)])
                batch_z = np.concatenate([batch_z, z], axis=0)
            batch_z = np.concatenate([batch_z, z2], axis=0)

            img = self.sess.run(self.fake_A,
                                feed_dict={self.z: batch_z, self.y: yp})

            save_images(img, [int(np.sqrt(self.batch_size)), int(np.sqrt(self.batch_size))],
                        './{}/test_G_{:2d}'.format(args.test_dir, k))

def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high
