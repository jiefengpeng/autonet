#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import numpy as np
import tensorflow as tf
import argparse
import os


from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorflow.python.training import moving_averages
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.tfutils.varreplace import freeze_variables

from triplet_loss import batch_all_triplet_loss

from _binary_out_grad import _binary_out_grad
user_module = tf.load_op_library('./binary_out.so')
binary_out = user_module.binary_out

"""
CIFAR10 DenseNet example. See: http://arxiv.org/abs/1608.06993
Code is developed based on Yuxin Wu's ResNet implementation: https://github.com/ppwwyyxx/tensorpack/tree/master/examples/ResNet
Results using DenseNet (L=40, K=12) on Cifar10 with data augmentation: ~5.77% test error.

Running time:
On one TITAN X GPU (CUDA 7.5 and cudnn 5.1), the code should run ~5iters/s on a batch size 64.
"""

BATCH_SIZE = 64

class Model(ModelDesc):
    def __init__(self, depth):
        super(Model, self).__init__()
        self.N = int((depth - 4)  / 3)
        self.growthRate = 12

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
      
        def get_mask(l, channel):
            moving_mask = tf.get_variable('mask_con/EMA', [channel],
                                          initializer=tf.constant_initializer(1.),
                                          trainable=False)
            ctx = get_current_tower_context()
            if ctx.is_training:
                m = GlobalAvgPooling('global_m', l) 
                m = FullyConnected('fc_con_1', m, out_dim=channel*2, nl=tf.nn.relu,) 
                                   #use_bias=False) 
                m = FullyConnected('fc_con_2', m, out_dim=channel)#, use_bias=False) 

                #m = Conv2D('conv_m', l, 1, 3, stride=1, use_bias=False, nl=tf.identity)
                #m = LayerNorm('ln_m', m) # N*H*W*1
                #m = tf.image.resize_images(m, [KERNAL_SIZE,KERNAL_SIZE]) # N*ks*ks*1
                m = tf.reduce_mean(m, 0) # C
                m = binary_out(m)
                #m = 1 / ( 1 + tf.exp(-10 * m))
                update_mask = moving_averages.assign_moving_average(moving_mask, m, 0.9,
                                  zero_debias=False, name='mask_con_ema_op')
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mask)
            else:
                m = moving_mask
            if ctx.is_main_training_tower:
                tf.contrib.framework.add_model_variable(moving_mask)

            m = tf.stack([m]*self.growthRate, 0)
            m = tf.transpose(m, [1,0])
            m = tf.reshape(m, [1,1,channel*self.growthRate,1]) 
            return m
            
        def add_layer(name, l):
            in_shape = l.get_shape().as_list()
            in_channel = in_shape[3]
            with tf.variable_scope(name) as scope:
                c = BatchNorm('bn1', l)
                c = tf.nn.relu(c)
                out_channel = self.growthRate
                filter_shape = [3, 3, in_channel, out_channel]
                stride = [1, 1, 1, 1]
                W_init = tf.random_normal_initializer(stddev=np.sqrt(2.0/9/out_channel))
                W = tf.get_variable('conv1/W', filter_shape, initializer=W_init)

                m = get_mask(c, in_channel // self.growthRate)

                W = W * m  # H*W*C_i*C_o
                c = tf.nn.conv2d(c, W, stride, 'SAME', data_format='NHWC')
                l = tf.concat([c, l], 3)
            return l

        def add_transition(name, l):
            shape = l.get_shape().as_list()
            in_channel = shape[3]
            group = 1#in_channel / self.growthRate
            with tf.variable_scope(name) as scope:
                l = BatchNorm('bn1', l)
                l = tf.nn.relu(l)
                l = Conv2D('conv1', l, in_channel, 1, stride=1, use_bias=False, nl=tf.nn.relu, split=group)
                l = AvgPooling('pool', l, 2)
                #l = MaxPooling('pool', l, 2)
            return l


        def dense_net(name):
            l = conv('conv0', image, self.growthRate, 1)
            with tf.variable_scope('block1') as scope:

                with tf.variable_scope('dense_layer.0'):
                    c = BatchNorm('bn1', l)
                    c = tf.nn.relu(c)
		    c = conv('conv1', c, self.growthRate, 1)
		    l = tf.concat([c, l], 3)

                for i in range(1, self.N):
                    l = add_layer('dense_layer.{}'.format(i), l)
                l = add_transition('transition1', l)

            with tf.variable_scope('block2') as scope:

                for i in range(self.N):
                    l = add_layer('dense_layer.{}'.format(i), l)
                l = add_transition('transition2', l)

            with tf.variable_scope('block3') as scope:

                for i in range(self.N):
                    l = add_layer('dense_layer.{}'.format(i), l)

            l = BatchNorm('bnlast', l)
            l = tf.nn.relu(l)
            l = GlobalAvgPooling('gap', l)

            logits = FullyConnected('linear', l, out_dim=10, nl=tf.identity)

            return logits, l

        logits, embedings = dense_net("dense_net")

        prob = tf.nn.softmax(logits, name='output')

        #cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        #cost = tf.reduce_mean(cost, name='cross_entropy_loss')
        #triplet_cost, fraction = batch_all_triplet_loss(label, embedings, margin=0.5, squared=False)
        cost, fraction = batch_all_triplet_loss(label, embedings, margin=0.5, squared=False)


        wrong = prediction_incorrect(logits, label)
        # monitor training error
        add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        # weight decay on all W
        wd_cost = tf.multiply(1e-4, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        #add_moving_summary(cost, triplet_cost, wd_cost)
        add_moving_summary(cost, wd_cost)
        add_moving_summary(fraction)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        #self.cost = tf.add_n([cost, triplet_cost, wd_cost], name='cost')
        self.cost = tf.add_n([cost, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        tf.summary.scalar('learning_rate', lr)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)


def get_data(train_or_test):
    isTrain = train_or_test == 'train'
    ds = dataset.Cifar10(train_or_test)
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
    log_dir = 'train_log/cifar10-triplet_loss-depth%s' % (str(args.depth))
    logger.set_logger_dir(log_dir, action='n')
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
        #TensorPrinter(['block1/dense_layer.{}/mask_con/EMA:0'.format(i) for i in xrange(1,N)] + ['block2/dense_layer.{}/mask_con/EMA:0'.format(j) for j in xrange(N)] + ['block3/dense_layer.{}/mask_con/EMA:0'.format(k) for k in xrange(N)])
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
    parser.add_argument('--depth',default=40, help='The depth of densenet')
    parser.add_argument('--max_epoch',default=300,help='max epoch')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    nr_tower = 1
    if args.gpu:
        nr_tower = len(args.gpu.split(','))

    N = int((args.depth - 4)  / 3)
    config = get_config()

    ignore = []
    #ignore = ['block1/dense_layer.{}/W_m:0'.format(i) for i in xrange(N)] + ['block2/dense_layer.{}/W_m:0'.format(i) for i in xrange(N)] + ['block3/dense_layer.{}/W_m:0'.format(i) for i in xrange(N)] + ['block1/dense_layer.{}/mask/EMA:0'.format(i) for i in xrange(N)] + ['block2/dense_layer.{}/mask/EMA:0'.format(i) for i in xrange(N)] + ['block3/dense_layer.{}/mask/EMA:0'.format(i) for i in xrange(N)] 

    if args.load:
        config.session_init = SaverRestore(args.load, ignore=ignore)
    
    
    # SyncMultiGPUTrainer(config).train()
    launch_train_with_config(config, SyncMultiGPUTrainer(nr_tower))#, scale=0.5))
