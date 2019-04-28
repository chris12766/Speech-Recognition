import numpy as np
import tensorflow as tf
import os


# directories

#main_dir = "C:\\Users\\chkar\\Desktop"   # LOCAL

main_dir = "/scratch/chk1g16"              # PUTTY
data_dir = os.path.join(main_dir, "speech_datasets")

# params
batch_size = 32
dropout_keep_prob_train = 0.5


def conv2d_relu(input, conv_kernel_shape, conv_padding='SAME', relu=True):
    # create conv filter from random normal distr with mean 0 and std dev 0.01
    filter = tf.Variable(tf.truncated_normal(shape=conv_kernel_shape,
                                             stddev=0.01),
                                             name='conv2d_kernel')
    net = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding=conv_padding)
    if relu:
        net = tf.nn.relu(net)
        
    return net

def conv_net_part(input, batch_norm_train_mode):
    with tf.name_scope('conv_net_part'):
        input = tf.layers.batch_normalization(input)
        # (batch_size, num_frames, num_mel_spec_bins) -> (batch_size, num_frames, num_mel_spec_bins, 1)
        # immitates image with (batch_size, height, width, num_channels)
        # (?, 112, 46, 1)
        input = tf.expand_dims(input, -1)
    
    
        # each conv kernel shape is [filter_height, filter_width, in_channels, out_channels]
        # Block 1
        net = conv2d_relu(input=input, conv_kernel_shape=[7, 7, 1, 16], 
                          conv_padding='VALID', relu=True)
        net = tf.layers.batch_normalization(net, training=batch_norm_train_mode)
        
        print(1)
        print(net.shape)
        print()
        
        # Block 2
        net = conv2d_relu(input=net, conv_kernel_shape=[5, 5, 16, 32], 
                          conv_padding='VALID', relu=True)
        net = tf.layers.batch_normalization(net, training=batch_norm_train_mode)
        
        print(2)
        print(net.shape)
        print()
          
        net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 1, 2, 1], padding='SAME')
        print(21)
        print(net.shape)
        print()
        
        # Block 3
        net = conv2d_relu(input=net, conv_kernel_shape=[3, 3, 32, 32], 
                          conv_padding='VALID', relu=True)
        net = tf.layers.batch_normalization(net, training=batch_norm_train_mode)
         
        print(3)
        print(net.shape)
        print()
        
        
        net = tf.nn.max_pool(value=net, ksize=[1, 2, 4, 1], strides=[1, 1, 4, 1], padding='SAME')
        
        print(31)
        print(net.shape)
        print()
        
        # Block 4
        net = conv2d_relu(input=net, conv_kernel_shape=[3, 3, 32, 32], 
                          conv_padding='VALID', relu=True)
        net = tf.layers.batch_normalization(net, training=batch_norm_train_mode)
        
        print(4)
        print(net.shape)
        print()
        
        net = tf.nn.max_pool(value=net, ksize=[1, 2, 4, 1], strides=[1, 1, 4, 1], padding='SAME')
        
        print(net.shape)
        '''
        1
        (?, 106, 235, 16)

        2
        (?, 102, 231, 32)

        21
        (?, 102, 116, 32)

        3
        (?, 100, 114, 32)

        31
        (?, 100, 29, 32)

        4
        (?, 98, 27, 32)

        (?, 98, 7, 32)
        '''
    return net

    
def conv_lstm_net(input, dropout_keep_prob, batch_norm_train_mode, num_classes):
    # conv part
    # input: (batch_size=?, 112, 46)
    # ouput: (batch_size=?, 98, 7, 32)
    conv_net_output = conv_net_part(input, batch_norm_train_mode)

    # rnn part
    with tf.name_scope('lstm_net_part'):
        # convert to shape: (batch_size=?, data_seq_len, num_feats_per_seq_fragment)
        data_seq_len = conv_net_output.shape[1]
        num_features_per_seq_fragment = conv_net_output.shape[2] * conv_net_output.shape[3]
        lstm_input = tf.reshape(conv_net_output, [-1, data_seq_len, num_features_per_seq_fragment])
        
        # make it time-major
        lstm_input = tf.transpose(lstm_input, [1, 0, 2])
        
        # create GRU cell and layer
        # output: (98, ?, 128)
        gru_output, state = tf.contrib.cudnn_rnn.CudnnGRU(num_layers=1,
                                                num_units=128,
                                                dropout=0.0,
                                                seed=None,
                                                dtype=tf.dtypes.float32,
                                                kernel_initializer=None,
                                                bias_initializer=None)(lstm_input)
                                                
                                                
        with tf.name_scope('fc_net_part'):
            batch_norm = lambda x : tf.layers.batch_normalization(x, training=batch_norm_train_mode)
        
            fc_net = tf.transpose(gru_output, [1, 0, 2])
            fc_net = tf.reshape(fc_net, [-1, fc_net.shape[1] * fc_net.shape[2]])
            
            '''
            # FC Block 1
            # BN and dropout
            fc_net = tf.layers.batch_normalization(fc_net, training=batch_norm_train_mode)
            if dropout_keep_prob != 1:
                fc_net = tf.nn.dropout(fc_net, keep_prob=dropout_keep_prob)
            
            fc_net = tf.contrib.layers.fully_connected(fc_net,
                                            num_outputs=128,
                                            activation_fn=tf.nn.relu,
                                            normalizer_fn=batch_norm,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            weights_regularizer=None)
            '''
            # FC Block 2
            if dropout_keep_prob != 1:
                fc_net = tf.nn.dropout(fc_net, keep_prob=dropout_keep_prob)
            
            # classification
            logits = tf.contrib.layers.fully_connected(fc_net,
                                            num_outputs=num_classes,
                                            # activation_fn=tf.nn.softmax,  according to tensorflow the loss function does it
                                            normalizer_fn=batch_norm,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            weights_regularizer=None)
                                            
    return logits
    

def create_train_graph(num_classes, num_frames, num_mel_spec_bins, init_lr, lr_decay_steps, lr_decay_rate):
    # batch placeholders
    # batch size is None as it is not necessarily the same all the time
    data_batch_plh = tf.placeholder(tf.float32, [None, num_frames, num_mel_spec_bins], name="data")
    label_batch_plh = tf.placeholder(tf.int32, [None], name="labels")
    
    # training placeholders
    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=(), name="dropout_keep_prob")
    batch_norm_train_mode = tf.placeholder(tf.bool, name='batch_norm_train_mode')

    # inference
    logits = conv_lstm_net(data_batch_plh, dropout_keep_prob, batch_norm_train_mode, num_classes)
    probabilities = tf.nn.softmax(logits)
    pred_values, pred_indices = tf.nn.top_k(probabilities, k=1)
    
    # loss and performance metrics
    loss = get_sparse_crossentropy_loss(logits, label_batch_plh)

    # optimizer
    train_op, global_step, learn_rate, global_norm = optimize_loss(loss, init_lr, lr_decay_steps, lr_decay_rate)

    
    summaries = tf.summary.merge_all()

    # arguments to sess.run()
    ops_to_run = [summaries, global_step, loss]
    
    train_feed_dict = {dropout_keep_prob: dropout_keep_prob_train,
                       batch_norm_train_mode: True}
    val_feed_dict = {batch_norm_train_mode: False}
    
    return (ops_to_run + [train_op, learn_rate, global_norm, pred_values, pred_indices], train_feed_dict), (ops_to_run + [pred_values, pred_indices], val_feed_dict), data_batch_plh, label_batch_plh



def get_sparse_crossentropy_loss(logits, label_batch):
    with tf.name_scope('loss'):
        # calculate the loss                                
        # labels [batch_size]
        # logits [batch_size, num_classes]
        crossentropy_loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_batch,
                                                                              logits=logits)
        # average over the batch
        loss = tf.reduce_mean(crossentropy_loss_op)

    tf.summary.scalar('crossentropy_loss', loss)

    return loss


def optimize_loss(loss, init_lr, lr_decay_steps, lr_decay_rate):
    global_step = tf.train.get_or_create_global_step()
    train_vars = tf.trainable_variables()

    with tf.name_scope('optimizer'):
        max_grad_norm = 10.0
        # adaptive learning rate
        learn_rate = tf.train.exponential_decay(learning_rate=init_lr, 
                                                global_step=global_step, 
                                                decay_steps=lr_decay_steps, 
                                                decay_rate=lr_decay_rate, 
                                                staircase=True,
                                                name='adapt_learn_rate')
        optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate,
                                           beta1=0.9,
                                           beta2=0.999,
                                           epsilon=1e-8,
                                           use_locking=False,
                                           name='Adam')
        # compute gradients
        gradients = tf.gradients(loss, train_vars)
        # gradient clipping
        clipped_gradients, global_norm = tf.clip_by_global_norm(t_list=gradients, 
                                                                clip_norm=max_grad_norm,
                                                                use_norm=None)
        grad_vars = zip(clipped_gradients, train_vars)

        # make sure update ops such as batch normalization occur before the training
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            # perform training and increment global_step
            train_op = optimizer.apply_gradients(grad_vars, global_step=global_step)
            
            
    tf.summary.scalar('learn_rate', learn_rate)
    tf.summary.scalar('global_norm', global_norm)
    
    return train_op, global_step, learn_rate, global_norm






