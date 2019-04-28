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


def conv2d_relu_pool_batch_norm(input, conv_kernel_shape, training, conv_padding='SAME', relu=True, 
                                pool_kernel=[1, 2, 2, 1], pool_strides=[1, 2, 2, 1], pool_padding='SAME'):
    # create conv filter from random normal distr with mean 0 and std dev 0.01
    filter = tf.Variable(tf.truncated_normal(shape=conv_kernel_shape,
                                             stddev=0.01),
                                             name='conv2d_kernel')
    net = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding=conv_padding)
    if relu:
        net = tf.nn.relu(net)
        
    #net = tf.nn.max_pool(value=net, ksize=pool_kernel, strides=pool_strides, padding=pool_padding)
    net = tf.layers.batch_normalization(net, training=training)
        
    return net

def conv_net_part(input, batch_norm_train_mode):
    with tf.name_scope('conv_net_part'):
        input = tf.layers.batch_normalization(input, training=batch_norm_train_mode)
        # (batch_size, num_frames, num_mel_spec_bins) -> (batch_size, num_frames, num_mel_spec_bins, 1)
        # immitates image with (batch_size, height, width, num_channels)
        # (?, 112, 46, 1)
        input = tf.expand_dims(input, -1)
    
    
        # each conv kernel shape is [filter_height, filter_width, in_channels, out_channels]
        # Block 1
        net = conv2d_relu_pool_batch_norm(input=input, conv_kernel_shape=[3, 3, 1, 32], 
                                          training=batch_norm_train_mode, conv_padding='VALID', relu=True)
        
        # Block 2
        net = conv2d_relu_pool_batch_norm(input=input, conv_kernel_shape=[3, 3, 32, 64], 
                                          training=batch_norm_train_mode, conv_padding='VALID', relu=True)
                                          
        # Block 3
        net = conv2d_relu_pool_batch_norm(input=input, conv_kernel_shape=[3, 3, 64, 128], 
                                          training=batch_norm_train_mode, conv_padding='VALID', relu=True)
                                          
        # Block 4
        net = conv2d_relu_pool_batch_norm(input=input, conv_kernel_shape=[3, 3, 128, 256], 
                                          training=batch_norm_train_mode, conv_padding='VALID', relu=True)
    
    
        print()
        print(net.shape)
        print()
        
        sys.exit()
    return net

    
def conv_lstm_net(input, dropout_keep_prob, batch_norm_train_mode, num_char_classes):
    # conv part
    # input: (batch_size=?, 112, 46)
    # ouput: (batch_size=?, 12, 4, 256)
    conv_net_output = conv_net_part(input, batch_norm_train_mode)

    # rnn part
    with tf.name_scope('lstm_net_part'):
        reverse_data_seqs = lambda x: tf.reverse(x, axis=[1])
        # convert to shape: (batch_size=?, data_seq_len, num_feats_per_seq_fragment)
        data_seq_len = conv_net_output.shape[1]
        num_features_per_seq_fragment = conv_net_output.shape[2] * conv_net_output.shape[3]
        lstm_input = tf.reshape(conv_net_output, [-1, data_seq_len, num_features_per_seq_fragment])
        
        # reverse the data sequences
        lstm_input = reverse_data_seqs(lstm_input)
        
        # create LSTM cell
        backward_lstm_cell = tf.nn.rnn_cell.LSTMCell(num_units=768,
                                        use_peepholes=True,
                                        cell_clip=None,
                                        initializer=None,
                                        num_proj=None,
                                        proj_clip=None,
                                        num_unit_shards=None,
                                        num_proj_shards=None,
                                        forget_bias=1.0,
                                        state_is_tuple=True,
                                        activation=None,
                                        reuse=None)
        # add the same dropout mask to the input and state at every step
        backward_lstm_cell_with_dropout = tf.nn.rnn_cell.DropoutWrapper(backward_lstm_cell, 
                                                    input_keep_prob=dropout_keep_prob, 
                                                    state_keep_prob=dropout_keep_prob, 
                                                    output_keep_prob=1.0,
                                                    variational_recurrent=True,  
                                                    input_size=num_features_per_seq_fragment,
                                                    dtype=tf.float32)
        
        # backwards LSTM layer
        backward_lstm_output, state = tf.nn.dynamic_rnn(backward_lstm_cell_with_dropout, 
                                                        lstm_input,
                                                        sequence_length=None,
                                                        time_major=False, 
                                                        scope='rnn', 
                                                        swap_memory=True,
                                                        parallel_iterations=8,
                                                        dtype=tf.float32)
        # reverse data sequences back to normal
        lstm_output = reverse_data_seqs(backward_lstm_output)
        # normalize
        # (batch_size=?/128, data_seq_len=12, features=768)
        lstm_net_output = tf.layers.batch_normalization(lstm_output, training=batch_norm_train_mode)


    # final fully-connected part
    with tf.name_scope('fc_net_part'):
        leaky_relu = lambda x: tf.nn.leaky_relu(x, alpha=0.75)
        batch_norm = lambda x: tf.layers.batch_normalization(x, training=batch_norm_train_mode)
        

        # FC Block 1
        if dropout_keep_prob != 1:
            fc_net = tf.nn.dropout(lstm_net_output, keep_prob=dropout_keep_prob)
        
        # use batch_norm instead of biases                                                                  
        # also use leaky_relu
        fc_net = tf.contrib.layers.fully_connected(fc_net,
                            num_outputs=160,
                            activation_fn=leaky_relu,
                            normalizer_fn=batch_norm,
                            normalizer_params=None,
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=None)
        # (batch_size=?, data_seq_len=12, features=160)
        fc_net = tf.nn.leaky_relu(fc_net, alpha=0.75)


        # FC Block 2
        if dropout_keep_prob != 1:
            fc_net = tf.nn.dropout(fc_net, keep_prob=dropout_keep_prob)
           
        # use batch_norm instead of biases
        # (batch_size=?, max_time/data_seq_len=12, num_char_classes=28)
        logits = tf.contrib.layers.fully_connected(fc_net,
                                num_outputs=num_char_classes,
                                activation_fn=None,
                                normalizer_fn=batch_norm,
                                normalizer_params=None,
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






