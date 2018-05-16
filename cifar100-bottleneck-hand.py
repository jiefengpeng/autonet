#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import argparse
import os
import h5py


from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorflow.python.training import moving_averages
from MyTrainer import MyTrainer
from tensorpack.utils.gpu import get_nr_gpu

"""
CIFAR10 DenseNet example. See: http://arxiv.org/abs/1608.06993
Code is developed based on Yuxin Wu's ResNet implementation: https://github.com/ppwwyyxx/tensorpack/tree/master/examples/ResNet
Results using DenseNet (L=40, K=12) on Cifar10 with data augmentation: ~5.77% test error.

Running time:
On one TITAN X GPU (CUDA 7.5 and cudnn 5.1), the code should run ~5iters/s on a batch size 64.
"""

BATCH_SIZE = 64
KERNEL = 5
BOTTLE_NUM = 2
LAYER_NUM = 10

class Model(ModelDesc):
    def __init__(self, depth):
        super(Model, self).__init__()
        self.N = BOTTLE_NUM #int((depth - 4)  / 3)
        #self.growthRate = 24

    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 32, 32, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')
               ]

    def _build_graph(self, input_vars):
        image, label = input_vars
        image = image / 128.0 - 1

        def conv(name, l, channel, stride):
            return Conv2D(name, l, channel, 3, stride=stride,
                          nl=tf.identity, use_bias=False,
                          W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0/9/channel)))
      
            
        def add_layer(name, inputs, mask, out_channel):
            input_tensors = [inputs[i] for i in mask['mask_con'][:]]
            c = tf.concat(input_tensors, 3)

            in_shape = c.get_shape().as_list()
            in_channel = in_shape[3]

            with tf.variable_scope(name) as scope:
                c = BatchNorm('bn1', c)
                c = tf.nn.relu(c)
        

                m_ker = mask['mask_ker'][:]

                filter_shape = [KERNEL, KERNEL, in_channel, out_channel]
                stride = [1, 1, 1, 1]
                W_init = tf.random_normal_initializer(stddev=np.sqrt(2.0/KERNEL**2/out_channel))
                W = tf.get_variable('conv1/W', filter_shape, initializer=W_init)
                W = W * m_ker  # H*W*C_i*C_o
                c = tf.nn.conv2d(c, W, stride, 'SAME', data_format='NHWC')
                l = [c] + inputs

            return l

        def add_transition(name, l, out_channel, pool=False):
            l = tf.concat(l, 3)

            with tf.variable_scope(name) as scope:
                l = BatchNorm('bn1', l)
                l = tf.nn.relu(l)
                l = Conv2D('conv1', l, out_channel, 1, stride=1, use_bias=False)
                if pool:
                    l = AvgPooling('pool', l, 2)
            return l


        def add_bottleneck(name, l, mask, out_channel, increase_dim=False, pool=False):
            in_shape = l.get_shape().as_list()
            in_channel = in_shape[3]
            short_cut = l
            with tf.variable_scope(name) as scope:
                l = BatchNorm('bn0', l)
                l = tf.nn.relu(l)
                c = Conv2D('conv0', l, out_channel, 1, stride=1, use_bias=False)
                c = [c]

                for i in range(LAYER_NUM):
                    c = add_layer('dense_layer.{}'.format(i), c, mask[str(i)], out_channel)

                #### avg & conv1
                #avg = AvgPooling('pool0', l, 3, strides=1, padding='SAME')
                #avg = Conv2D('pool_conv', avg, out_channel, 1, stride=1, use_bias=False)
                #conv1 = Conv2D('conv0', l, out_channel, 1, stride=1, use_bias=False)
                #c = [avg] + [conv1] + c
                

                c = add_transition('transition', c, out_channel*4, pool=(increase_dim and pool))

                if increase_dim:
                    if pool: short_cut = AvgPooling('residual_pool', short_cut, 2)
                if in_channel == out_channel*2:
                    short_cut = tf.pad(short_cut, [[0, 0], [0, 0], [0, 0], [in_channel//2, in_channel//2]])
                #if in_channel != out_channel*4:
                #    short_cut = Conv2D('shortcut_conv', l, out_channel*4, 1, stride=2 if pool else 1, use_bias=False)
                l = short_cut + c
            return l

        def dense_net(name):
            mask = h5py.File('N3L10K5.h5', 'r')
            l = conv('conv0', image, 64, 1)

            with tf.variable_scope('block1') as scope:

                for i in range(self.N):
                    l = add_bottleneck('dense_bottleneck.{}'.format(i), l, mask, 32, increase_dim=True if i == (self.N-1) else False, pool=True)

            with tf.variable_scope('block2') as scope:

                for i in range(self.N):
                    l = add_bottleneck('dense_bottleneck.{}'.format(i), l, mask, 64, increase_dim=True if i == (self.N-1) else False, pool=False)

            with tf.variable_scope('block3') as scope:

                for i in range(self.N):
                    l = add_bottleneck('dense_bottleneck.{}'.format(i), l, mask, 128,increase_dim=True if i == (self.N-1) else False, pool=True)

            with tf.variable_scope('block4') as scope:

                for i in range(self.N):
                    l = add_bottleneck('dense_bottleneck.{}'.format(i), l, mask, 256,increase_dim=True if i == (self.N-1) else False, pool=False)

            #with tf.variable_scope('block5') as scope:

            #    for i in range(self.N):
            #        l = add_bottleneck('dense_bottleneck.{}'.format(i), l, mask, 512,increase_dim=True if i == (self.N-1) else False, pool=False)

            l = BatchNorm('bnlast', l)
            l = tf.nn.relu(l)
            l = GlobalAvgPooling('gap', l)
            logits = FullyConnected('linear', l, out_dim=100, nl=tf.identity) 

            return logits

        logits = dense_net("dense_net")

        prob = tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = prediction_incorrect(logits, label)
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W
        wd_cost = tf.multiply(1e-4, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        tf.summary.scalar('learning_rate', lr)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar100(train_or_test)
    nr_tower = max(get_nr_gpu(), 1)
    pp_mean = ds.get_per_pixel_mean()
    if isTrain:
        augmentors = [
            imgaug.CenterPaste((40, 40)),
            imgaug.RandomCrop((32, 32)),
            imgaug.Flip(horiz=True),
            #imgaug.Brightness(20),
            #imgaug.Contrast((0.6,1.4)),
            imgaug.MapImage(lambda x: x - pp_mean),
        ]
    else:
        augmentors = [
            imgaug.MapImage(lambda x: x - pp_mean)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE//nr_tower, remainder=not isTrain)
    if isTrain:
        ds = PrefetchData(ds, 3, 2)
    return ds

def get_config():
    log_dir = 'train_log/cifar100-bottleneck-hand-N%s-L%s-K%s-B%s' % (str(BOTTLE_NUM), str(LAYER_NUM), str(KERNEL), '5')
    logger.set_logger_dir(log_dir)
    nr_tower = max(get_nr_gpu(), 1)

    # prepare dataset
    dataset_train = get_data('train')
    steps_per_epoch = dataset_train.size()
    dataset_test = get_data('test')

    callbacks=[
        ModelSaver(),
        #InferenceRunner(dataset_test,
        #    [ScalarStats('cost'), ClassificationError()]),
        ScheduledHyperParamSetter('learning_rate',
                                  [(0, 0.1), (args.drop_1, 0.01), (args.drop_2, 0.001)]),
        #TensorPrinter(['block1/dense_bottleneck.0/dense_layer.{}/mask_con/EMA:0'.format(i) for i in xrange(LAYER_NUM)] + ['block1/dense_bottleneck.0/dense_layer.{}/mask_ker/EMA:0'.format(j) for j in xrange(LAYER_NUM)])# + ['block3/dense_bottleneck.0/dense_layer.{}/mask_con/EMA:0'.format(k) for k in xrange(10)])
    ]
    if nr_tower == 1:
        callbacks.append(InferenceRunner(dataset_test,
	    [ScalarStats('cost'), ClassificationError()]))
    else:
        callbacks.append(DataParallelInferenceRunner(dataset_test,
	    [ScalarStats('cost'), ClassificationError()], list(range(nr_tower))))

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=callbacks,
        model=Model(depth=args.depth),
        steps_per_epoch=steps_per_epoch // nr_tower,
        max_epoch=args.max_epoch,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    parser.add_argument('--load', help='load model')
    parser.add_argument('--drop_1',default=150, help='Epoch to drop learning rate to 0.01.') # nargs='*' in multi mode
    parser.add_argument('--drop_2',default=225,help='Epoch to drop learning rate to 0.001')
    parser.add_argument('--depth',default=160, help='The depth of densenet')
    parser.add_argument('--max_epoch',default=300,help='max epoch')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


    nr_tower = 0
    if args.gpu:
        nr_tower = len(args.gpu.split(','))

    N = int((args.depth - 4)  / 3)
    config = get_config()

    ignore = ['global_step']
    #ignore = ['block1/dense_layer.{}/W_m:0'.format(i) for i in xrange(N)] + ['block2/dense_layer.{}/W_m:0'.format(i) for i in xrange(N)] + ['block3/dense_layer.{}/W_m:0'.format(i) for i in xrange(N)] + ['block1/dense_layer.{}/mask/EMA:0'.format(i) for i in xrange(N)] + ['block2/dense_layer.{}/mask/EMA:0'.format(i) for i in xrange(N)] + ['block3/dense_layer.{}/mask/EMA:0'.format(i) for i in xrange(N)] 

    if args.load:
        config.session_init = SaverRestore(args.load, ignore=ignore)
    
    
    # SyncMultiGPUTrainer(config).train()
    launch_train_with_config(config, SyncMultiGPUTrainer(nr_tower))#, scale=0.5))
