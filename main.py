import argparse
import os
import scipy.misc
import numpy as np
import tensorflow as tf
from model import amgan

parser = argparse.ArgumentParser(description='')
parser.add_argument('--z_dim', dest='z_dim',type= int, default=100,help='z_dimention')
parser.add_argument('--epoch', dest='epoch',type= int, default=1000,help='epoch')
parser.add_argument('--batch_size',dest='batch_size',type= int, default=64, help = 'bactch_size')
parser.add_argument('--train_size',dest='train_size',type= int, default=1e8, help = 'train_size')
parser.add_argument('--fine_size',dest='fine_size',type =int, default= 32, help ='fine_size')
parser.add_argument('--ngf', dest='ngf', type=int, default=64, help = 'first num_channel of generator layer')
parser.add_argument('--ndf', dest='ndf', type=int, default=64, help='# of discri filters in first conv layer')
parser.add_argument('--inputA_nc', dest='inputA_nc', type=int, default=3, help='# of input image A channels')
parser.add_argument('--inputB_nc', dest='inputB_nc', type=int, default=1, help='# of input image B channels')
parser.add_argument('--lr', dest='lr', type=float, default=0.00002, help='initial learning rate for adam')
parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
parser.add_argument('--dataset_name', dest='dataset_name', default='svhn_mnist',help='name of dataset')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint1', help='models are saved here')
parser.add_argument('--sample_dir', dest='sample_dir', default='./sample1', help='samples are saved here')
parser.add_argument('--log_dir', dest='log_dir', default='./log1', help='logs are saved here')
parser.add_argument('--phase', dest='phase', default='train', help='train, test')
args=parser.parse_args()

def main(_):
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    with tf.Session() as sess:
        model = amgan(sess,args)
        model.train(args) if args.phase == 'train' \
            else model.test(args)


if __name__ == '__main__':
    tf.app.run()


# CUDA_VISIBLE_DEVICES=1 python main.py --phase=test L zeus 3  L1 grus 1
# tensorboard --port=6010 --logdir=./log