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

    
def att_RNN_net(input, dropout_keep_prob, batch_norm_train_mode, num_classes):
    batch_norm = lambda x: tf.layers.batch_normalization(x, training=batch_norm_train_mode)
    
    # normalize
    input = batch_norm(input)
    # expand dims for convolution to work
    input = tf.expand_dims(input, -1)
    
    net = tf.contrib.layers.conv2d(inputs=input,
                                    num_outputs=10,
                                    kernel_size=[5,1],
                                    stride=1,
                                    padding='SAME',
                                    data_format=None,
                                    rate=1,
                                    activation_fn=tf.nn.relu,
                                    normalizer_fn=None,
                                    normalizer_params=None,
                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                    weights_regularizer=None,
                                    biases_initializer=tf.zeros_initializer(),
                                    biases_regularizer=None)
    net = batch_norm(net)
    net = tf.contrib.layers.conv2d(inputs=net,
                                num_outputs=1,
                                kernel_size=[5,1],
                                stride=1,
                                padding='SAME',
                                data_format=None,
                                rate=1,
                                activation_fn=tf.nn.relu,
                                normalizer_fn=None,
                                normalizer_params=None,
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                weights_regularizer=None,
                                biases_initializer=tf.zeros_initializer(),
                                biases_regularizer=None)
    net = batch_norm(net)

    # reshape to (125, 80)
    net = tf.squeeze(net, axis=[-1])

    # [b_s, seq_len, vec_dim]
    net = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(64, return_sequences = True))(net)
    # [b_s, seq_len, vec_dim]
    net = tf.keras.layers.Bidirectional(tf.keras.layers.CuDNNLSTM(64, return_sequences = True))(net) 

    query = tf.contrib.layers.fully_connected(net[:,64], #[b_s, vec_dim]
                                        num_outputs=128,
                                        activation_fn=None,
                                        normalizer_fn=None,
                                        normalizer_params=None,
                                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                                        weights_regularizer=None)

    # dot product attention
    attr_scores = tf.keras.layers.Dot(axes=[1,2])([query, net]) 
    #[b_s, seq_len]
    attr_scores = tf.nn.softmax(attr_scores, name='attr_softmax') 

    # rescale sequence to [b_s, vec_dim]
    attr_vector = tf.keras.layers.Dot(axes=[1,1])([attr_scores, net])

    net = tf.contrib.layers.fully_connected(attr_vector,
                                            num_outputs=64,
                                            activation_fn=tf.nn.relu,
                                            normalizer_fn=None,
                                            normalizer_params=None,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            weights_regularizer=None)
    net = tf.contrib.layers.fully_connected(net,
                                            num_outputs=32,
                                            activation_fn=None,
                                            normalizer_fn=None,
                                            normalizer_params=None,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            weights_regularizer=None)

    logits = tf.contrib.layers.fully_connected(net,
                                            num_outputs=num_classes,
                                            # activation_fn=tf.nn.softmax,  according to tensorflow the loss function does it
                                            normalizer_fn=None,
                                            normalizer_params=None,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            weights_regularizer=None)
    return logits 
    

def create_train_graph(num_classes, num_frames, num_mel_spec_bins, init_lr, lr_decay_steps, lr_decay_rate):
    # batch placeholders
    # batch size is None as it is not necessarily the same all the time
    data_batch_plh = tf.placeholder(tf.float32, [None, num_frames, num_mel_spec_bins], name="data")
    label_batch_plh = tf.placeholder(tf.int32, [None, num_classes], name="labels")
    
    # training placeholders
    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=(), name="dropout_keep_prob")
    batch_norm_train_mode = tf.placeholder(tf.bool, name='batch_norm_train_mode')

    # inference
    logits = att_RNN_net(data_batch_plh, dropout_keep_prob, batch_norm_train_mode, num_classes)
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
    
    return (ops_to_run + [train_op, learn_rate, global_norm], train_feed_dict), (ops_to_run + [pred_values, pred_indices], val_feed_dict), data_batch_plh, label_batch_plh



def get_sparse_crossentropy_loss(logits, label_batch):
    with tf.name_scope('loss'):
        # calculate the loss                                
        # labels [batch_size]
        # logits [batch_size, num_classes]
        print("AA", label_batch.shape)
        crossentropy_loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_batch,
                                                                              logits=logits)
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






