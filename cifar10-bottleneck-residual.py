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
BOTTLE_NUM = 3
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
      
        def get_mask(l, in_channel, channel, mask_scope, reuse=True, i=None):

            #with tf.variable_scope(mask_scope, reuse=reuse):
            moving_m_con = tf.get_variable('mask_con/EMA', [channel],
                                          initializer=tf.constant_initializer(1.),
                                          trainable=False)
            moving_m_ker = tf.get_variable('mask_ker/EMA', [KERNEL,KERNEL,1,1],
                                          initializer=tf.constant_initializer(1.),
                                          trainable=False)
            moving_m_res = tf.get_variable('mask_res/EMA', [channel],
                                          initializer=tf.constant_initializer(1.),
                                          trainable=False)
            ctx = get_current_tower_context()
            if ctx.is_training:
                m = GlobalAvgPooling('global_m', l) 
                if i != 0:
	            # auto concat
                    m_con = FullyConnected('fc_con_1', m, out_dim=channel*2, nl=tf.nn.relu,) 
                    m_con = FullyConnected('fc_con_2', m_con, out_dim=channel)
                    m_con = tf.reduce_mean(m_con, 0) # C
                    m_con = 1 / ( 1 + tf.exp(-10 * m_con))
                    update_m_con = moving_averages.assign_moving_average(moving_m_con, m_con, 0.99,
                                      zero_debias=False, name='mask_con_ema_op')
                    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_m_con)
	            # auto kernel
                    m_ker = FullyConnected('fc_ker_1', m, out_dim=KERNEL**2, nl=tf.nn.relu,) 
                    m_ker = FullyConnected('fc_ker_2', m_ker, out_dim=KERNEL**2)
                    m_ker = tf.reduce_mean(m_ker, 0) # C
                    m_ker = 1 / ( 1 + tf.exp(-10 * m_ker))
                    m_ker = tf.reshape(m_ker, [KERNEL,KERNEL,1,1]) 
                    update_m_ker = moving_averages.assign_moving_average(moving_m_ker, m_ker, 0.99,
                                      zero_debias=False, name='mask_ker_ema_op')
                    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_m_ker)
                else:
                    m_con = moving_m_con

                    # auto kernel
                    m_ker = FullyConnected('fc_ker_1', m, out_dim=KERNEL**2, nl=tf.nn.relu,) 
                    m_ker = FullyConnected('fc_ker_2', m_ker, out_dim=KERNEL**2)
                    m_ker = tf.reduce_mean(m_ker, 0) # C
                    m_ker = 1 / ( 1 + tf.exp(-10 * m_ker))
                    m_ker = tf.reshape(m_ker, [KERNEL,KERNEL,1,1]) 
                    update_m_ker = moving_averages.assign_moving_average(moving_m_ker, m_ker, 0.99,
                                      zero_debias=False, name='mask_ker_ema_op')
                    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_m_ker)

                # auto residual
                m_res = FullyConnected('fc_res_1', m, out_dim=channel*2, nl=tf.nn.relu,) 
                m_res = FullyConnected('fc_res_2', m_res, out_dim=channel)
                m_res = tf.reduce_mean(m_res, 0) # C
                m_res = 1 / ( 1 + tf.exp(-10 * m_res))
                update_m_res = moving_averages.assign_moving_average(moving_m_res, m_res, 0.99,
                                  zero_debias=False, name='mask_res_ema_op')
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_m_res)
            else:
                m_con = moving_m_con
                m_ker = moving_m_ker
                m_res = moving_m_res

            #m_con = tf.stack([m_con]*(in_channel//channel), 0)
            #m_con = tf.transpose(m_con, [1,0])
            #m_con = tf.reshape(m_con, [1,1,in_channel,1]) 

            if ctx.is_main_training_tower:
                tf.contrib.framework.add_model_variable(moving_m_con)
                tf.contrib.framework.add_model_variable(moving_m_ker)
                tf.contrib.framework.add_model_variable(moving_m_res)

            return m_con, m_ker, m_res
            
        def add_layer(name, l, store_con, store_ker, store_res, out_channel, mask_scope, reuse, i=None):
            in_shape = l.get_shape().as_list()
            in_channel = in_shape[3]
            #with tf.variable_scope(mask_scope):
            #    with tf.variable_scope(name, reuse=reuse) as mask_scope:
            #        assert mask_scope.name == 'MASK/'+name, '{} vs {}'.format(mask_scope.name, 'MASK/'+name) 

            with tf.variable_scope(name) as scope:
                c = BatchNorm('bn1', l)
                c = tf.nn.relu(c)
        
                if not reuse:
                    m_con, m_ker, m_res = get_mask(c, in_channel, in_channel // out_channel, mask_scope, reuse, i)
                    store_con.append(m_con)
                    store_ker.append(m_ker)
                    store_res.append(m_res)
                else:
                    m_con = store_con[i]
                    m_ker = store_ker[i]
                    m_res = store_res[i]

                m_con = tf.stack([m_con]*out_channel, 0)
                m_con = tf.transpose(m_con, [1,0])
                m_con = tf.reshape(m_con, [1,1,in_channel,1]) 
                m_res = tf.stack([m_res]*out_channel, 0)
                m_res = tf.transpose(m_res, [1,0])
                m_res = tf.reshape(m_res, [1,1,1,in_channel]) 

                filter_shape = [KERNEL, KERNEL, in_channel, out_channel]
                stride = [1, 1, 1, 1]
                W_init = tf.random_normal_initializer(stddev=np.sqrt(2.0/KERNEL**2/out_channel))
                W = tf.get_variable('conv1/W', filter_shape, initializer=W_init)
                W = W * m_con  # H*W*C_i*C_o
                W = W * m_ker  # H*W*C_i*C_o
                c = tf.nn.conv2d(c, W, stride, 'SAME', data_format='NHWC')

                res = l * m_res
                res = tf.add_n(tf.split(res, in_channel//out_channel, 3))
                c = c + res

                l = tf.concat([c, l], 3)
            return l, store_con, store_ker, store_res

        def add_transition(name, l, out_channel, pool=False):
            shape = l.get_shape().as_list()
            #in_channel = shape[3]
            with tf.variable_scope(name) as scope:
                l = BatchNorm('bn1', l)
                l = tf.nn.relu(l)
                l = Conv2D('conv1', l, out_channel, 1, stride=1, use_bias=False)
                if pool:
                    l = AvgPooling('pool', l, 2)
            return l


        def add_bottleneck(name, l, store_con, store_ker, store_res, out_channel, mask_scope, reuse, increase_dim=False, pool=False):
            in_shape = l.get_shape().as_list()
            in_channel = in_shape[3]
            with tf.variable_scope(name) as scope:
                c = l

                for i in range(LAYER_NUM):
                    c, store_con, store_ker, store_res = add_layer('dense_layer.{}'.format(i), c, store_con, store_ker, store_res, out_channel, mask_scope, reuse, i)

                if increase_dim: out_channel *= 2
                c = add_transition('transition', c, out_channel, pool=(increase_dim and pool))

                if increase_dim:
                    if pool: l = AvgPooling('residual_pool', l, 2)
                    l = tf.pad(l, [[0, 0], [0, 0], [0, 0], [in_channel // 2, in_channel // 2]])
                l = l + c
            return l

        def dense_net(name):
            with tf.variable_scope('MASK') as MASK_scope:
                assert MASK_scope.name == 'MASK'

            l = conv('conv0', image, 16, 1)
            store_con = []
            store_ker = []
            store_res = []
            with tf.variable_scope('block1') as scope:

                with tf.variable_scope('dense_bottleneck.0'):
                    for i in range(LAYER_NUM):
                        l, store_con, store_ker, store_res = add_layer('dense_layer.{}'.format(i), l, store_con, store_ker, store_res, 16, MASK_scope, reuse=False, i=i)
                    l = add_transition('transition', l, 16)

                for i in range(1, self.N):
                    l = add_bottleneck('dense_bottleneck.{}'.format(i), l, store_con, store_ker, store_res, 16, MASK_scope, reuse=True, increase_dim=True if i == (self.N-1) else False, pool=True)

            with tf.variable_scope('block2') as scope:

                for i in range(self.N):
                    l = add_bottleneck('dense_bottleneck.{}'.format(i), l, store_con, store_ker, store_res, 32, MASK_scope, reuse=True, increase_dim=True if i == (self.N-1) else False, pool=False)

            with tf.variable_scope('block3') as scope:

                for i in range(self.N):
                    l = add_bottleneck('dense_bottleneck.{}'.format(i), l, store_con, store_ker, store_res, 64, MASK_scope, reuse=True, increase_dim=True if i == (self.N-1) else False, pool=True)

            with tf.variable_scope('block4') as scope:

                for i in range(self.N):
                    l = add_bottleneck('dense_bottleneck.{}'.format(i), l, store_con, store_ker, store_res, 128, MASK_scope, reuse=True,increase_dim=True if i == (self.N-1) else False, pool=False)

            l = BatchNorm('bnlast', l)
            l = tf.nn.relu(l)
            l = GlobalAvgPooling('gap', l)
            logits = FullyConnected('linear', l, out_dim=10, nl=tf.identity) 

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
    log_dir = 'train_log/cifar10-bottleneck-residual-N%s-L%s-K%s' % (str(BOTTLE_NUM), str(LAYER_NUM), str(KERNEL))
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
                                  [(1, 0.1), (args.drop_1, 0.01), (args.drop_2, 0.001)]),
        #TensorPrinter(['block1/dense_bottleneck.0/dense_layer.{}/mask_con/EMA:0'.format(i) for i in xrange(LAYER_NUM)]) #+ ['block2/dense_bottleneck.0/dense_layer.{}/mask_con/EMA:0'.format(j) for j in xrange(10)] + ['block3/dense_bottleneck.0/dense_layer.{}/mask_con/EMA:0'.format(k) for k in xrange(10)])
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
