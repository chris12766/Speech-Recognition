import numpy as np
import tensorflow as tf
import os


# directories

#main_dir = "C:\\Users\\chkar\\Desktop"   # LOCAL


main_dir = "/scratch/chk1g16"              # PUTTY


saves_dir = os.path.join(main_dir, "speech_project_saves")
data_dir = os.path.join(main_dir, "speech_datasets")

if not os.path.isdir(saves_dir):
    os.mkdir(saves_dir)
log_dir = os.path.join(saves_dir, "logs")
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
ckpt_dir = os.path.join(saves_dir, "ckpts")
if not os.path.isdir(ckpt_dir):
    os.mkdir(ckpt_dir)


# Training params
batch_size = 128
num_epochs = 1
init_lr = 0.0002
lr_decay_steps = 0.3
dropout_keep_prob_train = 0.5
# num_batches/lr_decay_rate = 2.3
lr_decay_rate = 3800



def conv2d_batch_norm_relu(input, kernel_shape, training, padding='SAME', relu=True):
    # create conv filter from random normal distr with mean 0 and std dev 0.01
    filter = tf.Variable(tf.truncated_normal(shape=kernel_shape,
                                             stddev=0.01),
                                             name='conv2d_kernel')
    net = tf.nn.conv2d(input, filter, strides=[1, 1, 1, 1], padding=padding)

    net = tf.layers.batch_normalization(net, training=training)
    if relu:
        net = tf.nn.relu(net)
    return net


def conv_net_part(input, batch_norm_train_mode):
    with tf.name_scope('conv_net_part'):
        input = tf.layers.batch_normalization(input, training=batch_norm_train_mode)
        # (batch_size, num_frames, num_mel_spec_bins) -> (batch_size, num_frames, num_mel_spec_bins, 1)
        # immitates image with (batch_size, height, width, num_channels)
        # (?, 112, 46, 1)
        input = tf.expand_dims(input, -1)
    
    
        # each kernel shape is [filter_height, filter_width, in_channels, out_channels]
        
        # Block 1
        net = conv2d_batch_norm_relu(input=input, kernel_shape=[5, 3, 1, 64], training=batch_norm_train_mode, padding='VALID', relu=True)
        
        residual = net
        # conv2d_batch_norm_relu x 2 + res -> RELU
        net = conv2d_batch_norm_relu(net, kernel_shape=[3, 3, 64, 64], training=batch_norm_train_mode, padding='SAME', relu=True)
        net = conv2d_batch_norm_relu(net, kernel_shape=[3, 3, 64, 64], training=batch_norm_train_mode, padding='SAME', relu=False)
        net = net + residual
        net = tf.nn.relu(net)
        
       
        residual = net
        # conv2d_batch_norm_relu x 2 + res -> RELU
        net = conv2d_batch_norm_relu(net, kernel_shape=[3, 3, 64, 64], training=batch_norm_train_mode, padding='SAME', relu=True)
        net = conv2d_batch_norm_relu(net, kernel_shape=[3, 3, 64, 64], training=batch_norm_train_mode, padding='SAME', relu=False)
        net = net + residual
        net = tf.nn.relu(net)
        
        net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        
        # Block 2
        net = conv2d_batch_norm_relu(input=net, kernel_shape=[3, 3, 64, 128], training=batch_norm_train_mode, padding='VALID', relu=True)
        
        residual = net
        # conv2d_batch_norm_relu x 2 + res -> RELU
        net = conv2d_batch_norm_relu(net, kernel_shape=[3, 3, 128, 128], training=batch_norm_train_mode, padding='SAME', relu=True)
        net = conv2d_batch_norm_relu(net, kernel_shape=[3, 3, 128, 128], training=batch_norm_train_mode, padding='SAME', relu=False)
        net = net + residual
        net = tf.nn.relu(net)
        
        residual = net
        # conv2d_batch_norm_relu x 2 + res -> RELU
        net = conv2d_batch_norm_relu(net, kernel_shape=[3, 3, 128, 128], training=batch_norm_train_mode, padding='SAME', relu=True)
        net = conv2d_batch_norm_relu(net, kernel_shape=[3, 3, 128, 128], training=batch_norm_train_mode, padding='SAME', relu=False)
        net = net + residual
        net = tf.nn.relu(net)
        
        net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
    
        # Block 3
        net = conv2d_batch_norm_relu(input=net, kernel_shape=[3, 3, 128, 256], training=batch_norm_train_mode, padding='VALID', relu=True)
        
        residual = net
        # conv2d_batch_norm_relu x 2 + res -> RELU
        net = conv2d_batch_norm_relu(net, kernel_shape=[3, 3, 256, 256], training=batch_norm_train_mode, padding='SAME', relu=True)
        net = conv2d_batch_norm_relu(net, kernel_shape=[3, 3, 256, 256], training=batch_norm_train_mode, padding='SAME', relu=False)
        net = net + residual
        net = tf.nn.relu(net)
        
        residual = net
        # conv2d_batch_norm_relu x 2 + res -> RELU
        net = conv2d_batch_norm_relu(net, kernel_shape=[3, 3, 256, 256], training=batch_norm_train_mode, padding='SAME', relu=True)
        net = conv2d_batch_norm_relu(net, kernel_shape=[3, 3, 256, 256], training=batch_norm_train_mode, padding='SAME', relu=False)
        net = net + residual
        net = tf.nn.relu(net)
        
        net = tf.nn.max_pool(value=net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
    return net

    
def conv_lstm_net(input, num_char_classes, dropout_keep_prob, batch_norm_train_mode):
    # conv part
    # input: (batch_size=?, data_seq_len=12, 4, 256)
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
        
        # use batch_norm instead of biases                                                                  # USE XAVIER INIT FOR ALL KERNELS!!!!!
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
        # (batch_size=?, data_seq_len=12, num_char_classes=28)
        logits = tf.contrib.layers.fully_connected(fc_net,
                                num_outputs=num_char_classes,
                                activation_fn=None,
                                normalizer_fn=batch_norm,
                                normalizer_params=None,
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                weights_regularizer=None)
                                
    return logits


def get_ctc_loss(logits, label_batch):
    # logits: [batch_size, max_time, num_classes]
    # 1-D tensor showing the length for each label in the batch
    batch_labels_lengths = tf.fill([tf.shape(label_batch)[0]], tf.shape(label_batch)[1])

    with tf.name_scope('loss'):
        # get sparse represenattion of the labels
        non_zero_elems_coords = tf.where(tf.not_equal(label_batch, 0))
        non_zero_elems = tf.gather_nd(label_batch, non_zero_elems_coords)
        sparse_label_batch = tf.SparseTensor(indices=non_zero_elems_coords, 
                                             values=non_zero_elems,
                                             dense_shape=tf.shape(label_batch, out_type=tf.int64))
                                        
        # calculate ctc loss                                
        ctc_loss_op = tf.nn.ctc_loss(labels=sparse_label_batch, 
                                     inputs=logits, 
                                     sequence_length=batch_labels_lengths,
                                     preprocess_collapse_repeated=True, 
                                     time_major=False, 
                                     ctc_merge_repeated=True,
                                     ignore_longer_outputs_than_inputs=False)
        loss = tf.reduce_mean(ctc_loss_op)
        
        
        prediction_probabilities = tf.nn.softmax(logits)
        max_probabilities = tf.reduce_max(prediction_probabilities, axis=2)
        raw_predictions = tf.argmax(prediction_probabilities, axis=2)



        # greedy decode logits
        # greedy decoder - beeam decoder with beam_width=1 and top_paths=1
        logits_T = tf.transpose(logits, perm=[1, 0, 2])
        greedy_predictions, neg_sum_logits = tf.nn.ctc_greedy_decoder(inputs=logits_T, 
                                                sequence_length=batch_labels_lengths,
                                                merge_repeated=True)
        
        # get greedy performance metrics
        edit_dist_greedy = tf.edit_distance(tf.cast(greedy_predictions[0], tf.int32),
                                            sparse_label_batch, 
                                            normalize=False)
        acc_greedy = tf.reduce_mean(tf.cast(tf.equal(edit_dist_greedy, 0), tf.float32))
        edit_dist_greedy = tf.reduce_mean(edit_dist_greedy)


        
        # beam decode logits
        beam_predictions, log_probabilities = tf.nn.ctc_beam_search_decoder(inputs=logits_T, 
                                                    sequence_length=batch_labels_lengths, 
                                                    beam_width=100,
                                                    top_paths=2,
                                                    merge_repeated=True)
                                        
        # get beam performance metrics
        edit_dist_beam = tf.edit_distance(tf.cast(beam_predictions[0], tf.int32),
                                              sparse_label_batch,
                                              normalize=False)
        acc_beam = tf.reduce_mean(tf.cast(tf.equal(edit_dist_beam, 0), tf.float32))
        
        
        predictions = tf.cast(tf.sparse.to_dense(beam_predictions[0]), tf.int32)
        scores = log_probabilities[:, 0] - log_probabilities[:, 1]

    tf.summary.scalar('ctc_loss', loss)
    tf.summary.scalar('acc_greedy', acc_greedy)
    tf.summary.scalar('edit_dist_greedy', edit_dist_greedy)
    tf.summary.scalar('confidence_score', tf.reduce_mean(scores))
    tf.summary.scalar('confidence_score', edit_dist_beam)
    tf.summary.scalar('confidence_score', acc_beam)

    return predictions, loss, acc_greedy, edit_dist_greedy, acc_beam, edit_dist_beam, scores


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
    
    return train_op, global_step

def create_train_graph(num_char_classes, label_encoding_length, num_frames, num_mel_spec_bins):
    # batch placeholders
    # batch size is None as it is not necessarily the same all the time
    data_batch_plh = tf.placeholder(tf.float32, [None, num_frames, num_mel_spec_bins], name="data")
    label_batch_plh = tf.placeholder(tf.int32, [None, label_encoding_length], name="label")
    
    # training placeholders
    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=(), name="dropout_keep_prob")
    batch_norm_train_mode = tf.placeholder(tf.bool, name='batch_norm_train_mode')

    # inference
    logits = conv_lstm_net(data_batch_plh, num_char_classes, dropout_keep_prob, batch_norm_train_mode)
    
    # loss and performance metrics
    predictions, loss, acc_greedy, edit_dist_greedy, acc_beam, edit_dist_beam, scores = get_ctc_loss(logits, label_batch_plh)

    # optimizer
    train_op, global_step = optimize_loss(loss, init_lr, lr_decay_steps, lr_decay_steps)

    
    summaries = tf.summary.merge_all()

    # arguments to sess.run()
    ops_to_run = [summaries, global_step, loss, acc_greedy, edit_dist_greedy, acc_beam, edit_dist_beam, scores]
    
    train_feed_dict = {dropout_keep_prob: dropout_keep_prob_train,
                       batch_norm_train_mode: True}
    val_feed_dict = {batch_norm_train_mode: False}
    
    return (ops_to_run + [train_op], train_feed_dict), (ops_to_run + [predictions], val_feed_dict), data_batch_plh, label_batch_plh


'''
def create_inference_graph(FLAGS, num_char_classes, label_encoding_length):
        audio_length = int(sampling_rate * audio_dur_in_ms / 1000)
        frame_size = int(sampling_rate * frame_size_ms / 1000)
        frame_stride = int(sampling_rate * frame_stride_ms / 1000)
        num_mel_spec_bins = num_mel_spec_bins
        tf.logging.info('audios_inference parameters: %s',
                                        [audio_length, frame_size, frame_stride, num_mel_spec_bins])

        audio = tf.placeholder(tf.float32, [None, audio_length],
                                                                 name='audio')
        spectrogram = convert_to_spectrogram(
                audio, sampling_rate, frame_size, frame_stride, num_mel_spec_bins)

        batch_norm_train_mode = tf.placeholder(tf.bool, name='batch_norm_train_mode')
        logits, dbg_layers, dbg_embeddings = conv_lstm_net(
                spectrogram, num_char_classes, dropout_keep_prob=1, batch_norm_train_mode=batch_norm_train_mode)
        prob = tf.nn.softmax(logits)
        chars = tf.argmax(prob, axis=2)

        label_lengths = tf.fill([tf.shape(spectrogram)[0]], label_encoding_length)
        logits_T = tf.transpose(logits, perm=[1, 0, 2])
        pred, score_cmp = tf.nn.ctc_beam_search_decoder(
                logits_T, label_lengths, top_paths=2)
        pred_cmp = [tf.cast(tf.sparse.to_dense(p), tf.int32)
                                            for p in pred]
        predicts = pred_cmp[0]
        scores = score_cmp[:, 0] - score_cmp[:, 1]
'''       
        















