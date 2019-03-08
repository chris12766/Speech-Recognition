import numpy as np
import tensorflow as tf
import os


# directories
saves_dir = "D:\\speech_project_saves"
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
dec_rate_lr = 0.3
dropout_keep_prob_const = 0.5
# num_batches/lr_dec_step = 2.3
lr_dec_step = 3800


# Data params
bg_nsr = 0.5
bg_noise_prob = 0.75
sampling_rate = 16000
frame_size_ms = 30.0
frame_stride_ms = 10.0
# fg_interp_factor = audio_dur_in_ms/(audio_dur_in_ms-padding_ms
padding_ms = 140
audio_dur_in_ms = 1140


# Model params
num_mel_spec_bins = 46
l2_scale = 0



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
        # (batch_size=?/128, 12, 768)
        lstm_net_output = tf.layers.batch_normalization(lstm_output, training=batch_norm_train_mode)


    # final fully-connected part
    with tf.name_scope('fc_net_part'):
        leaky_relu = lambda x: tf.nn.leaky_relu(x, alpha=0.75)
        batch_norm = lambda x: tf.layers.batch_normalization(x, training=batch_norm_train_mode)
        
        # reshape to 2d: (batch_size=128/?, features=rest/9216)
        fc_net = tf.reshape(lstm_net_output, [-1, lstm_net_output.shape[1] * lstm_net_output.shape[2]])

        # FC Block 1
        if dropout_keep_prob != 1:
            fc_net = tf.nn.dropout(fc_net, keep_prob=dropout_keep_prob)
        
        # use batch_norm instead of biases
        # also use leaky_relu
        fc_net = tf.contrib.layers.fully_connected(fc_net,
                            num_outputs=160,
                            activation_fn=leaky_relu,
                            normalizer_fn=batch_norm,
                            normalizer_params=None,
                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                            weights_regularizer=None)
        # (batch_size=?, 160)
        fc_net = tf.nn.leaky_relu(fc_net, alpha=0.75)

        # FC Block 2
        if dropout_keep_prob != 1:
            fc_net = tf.nn.dropout(fc_net, keep_prob=dropout_keep_prob)
            
        # use batch_norm instead of biases
        # (batch_size=?/128, 28)
        fc_net = tf.contrib.layers.fully_connected(fc_net,
                                num_outputs=num_char_classes,
                                activation_fn=None,
                                normalizer_fn=batch_norm,
                                normalizer_params=None,
                                weights_initializer=tf.contrib.layers.xavier_initializer(),
                                weights_regularizer=None)
                            
                            
        # return to 3d (batch_size, data_seq_len, pred  ictions)
        # (batch_size=?, data_seq_len=12, num_char_classes=28)
        logits = tf.reshape(fc_net, [-1, data_seq_len, num_char_classes])
    
        print()
        print()
        print(logits)
        print()
        print()
    
    return logits
    
def cal_perf(pred, sparse_labels):
    """Helper function to calculate edit distance and accuracy.
    """
    edist = tf.edit_distance(tf.cast(pred[0], tf.int32), sparse_labels,
                                                     normalize=False)
    acc = tf.reduce_mean(tf.cast(tf.equal(edist, 0), tf.float32))
    return edist, acc


def nn_cost_ctc(logits, labels, label_lengths):
    """Calculates network CTC cost, accuracy, predictions and confidence scores.
    """
    with tf.name_scope('loss_ctc'):
        idx = tf.where(tf.not_equal(labels, 0))
        sparse_labels = tf.SparseTensor(idx, tf.gather_nd(labels, idx),
                                        tf.shape(labels, out_type=tf.int64))
        loss = tf.reduce_mean(tf.nn.ctc_loss(
                              sparse_labels, logits, label_lengths,
                              preprocess_collapse_repeated=True, time_major=False))
        logits_transposed = tf.transpose(logits, perm=[1, 0, 2])
        probs = tf.nn.softmax(logits)
        max_probs = tf.reduce_max(probs, axis=2)
        raw_preds = tf.argmax(probs, axis=2)

    with tf.name_scope('acc_greedy'):
        pred_greedy, _ = tf.nn.ctc_greedy_decoder(logits_transposed, label_lengths)
        edist_greey, acc_greedy = cal_perf(pred_greedy, sparse_labels)
        edist_greey = tf.reduce_mean(edist_greey)

    with tf.name_scope('acc_beam'):
        pred_beam, prob_scores = tf.nn.ctc_beam_search_decoder(
                logits_transposed, label_lengths, top_paths=2)
        edist, acc_beam = cal_perf(pred_beam, sparse_labels)
        
        preds = tf.cast(tf.sparse.to_dense(pred_beam[0]), tf.int32)
        scores = prob_scores[:, 0] - prob_scores[:, 1]
        
        
        #sys.exit()
        
        
    tf.summary.scalar('ctc_loss', loss)
    tf.summary.scalar('acc_greedy', acc_greedy)
    tf.summary.scalar('edist_greey', edist_greey)
    # tf.summary.scalar('confidence_score', tf.reduce_mean(scores))

    return loss, acc_greedy, acc_beam, edist, preds, scores, raw_preds, max_probs


def nn_optimizer(loss, l2_scale, init_lr, dec_rate_lr, lr_dec_step, dbg=False):
    """Network cost optimizer, l2_regularizer, gradient clipping
        and batch_norm update.
    """
    global_step = tf.train.get_or_create_global_step()
    train_variables = tf.trainable_variables()

    sum_l2_loss = 0
    num_params = 0
    with tf.name_scope('regularizer'):
        for var in train_variables:
            if 'kernel' in var.op.name:
                num_params += np.prod(var.shape.as_list())
                v_loss = tf.nn.l2_loss(var)
                if dbg:
                    tf.summary.scalar(var.op.name + '/w_l2', v_loss)
                sum_l2_loss += v_loss
        loss += sum_l2_loss * l2_scale
        tf.logging.debug('num_weights_params=%d', num_params)

    with tf.name_scope('optimizer'):
        max_grad_norm = 10.0
        learning_rate = tf.train.exponential_decay(init_lr, global_step, dec_rate_lr, lr_dec_step, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
        grads = tf.gradients(loss, train_variables)
        mod_grads, ctc_global_norm = tf.clip_by_global_norm(grads, max_grad_norm)

        if dbg:
            for var in tf.global_variables():
                if 'batch_normalization' in var.op.name:
                    tf.summary.histogram(var.op.name, var)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        tf.logging.debug('optimizer dependencies: %d', len(update_ops))
        with tf.control_dependencies(update_ops):
            train_step = optimizer.apply_gradients(zip(mod_grads, train_variables),
                                                                                         global_step=global_step)

    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('l2loss', sum_l2_loss)
    tf.summary.scalar('global_norm', ctc_global_norm)

    tf.logging.debug('l2_scale=%f, max_grad_norm=%f', l2_scale, max_grad_norm)
    return train_step, global_step

def create_train_graph(num_char_classes, max_encoding_length):
    audio_length = int(sampling_rate * audio_dur_in_ms / 1000)
    frame_size = int(sampling_rate * frame_size_ms / 1000)
    frame_stride = int(sampling_rate * frame_stride_ms / 1000)

    # batch placeholders
    # batch size is None as it is not necessarily the same all the time
    data_batch = tf.placeholder(tf.float32, [None, audio_length], name="audio")
    label_batch = tf.placeholder(tf.int32, [None, max_encoding_length], name="label")
    
    # training placeholders
    dropout_keep_prob = tf.placeholder_with_default(1.0, shape=(), name="dropout_keep_prob")
    batch_norm_train_mode = tf.placeholder(tf.bool, name='batch_norm_train_mode')
    
    # convert audio to spectrograms
    # (batch_size, num_frames, num_mel_spec_bins)
    spectrograms = get_spectrograms(data_batch, sampling_rate, frame_size, frame_stride, num_mel_spec_bins)

    # perform inference
    logits = conv_lstm_net(spectrograms, num_char_classes, dropout_keep_prob, batch_norm_train_mode)
    

    print("WOOOO")


    sys.exit()
    
    # ctc_cost function
    label_lengths = tf.fill([tf.shape(spectrograms)[0]], max_encoding_length)
    loss, acc_fgreedy, acc_beam, edist, \
            predicts, scores, raw_preds, \
            max_probs = nn_cost_ctc(logits, label_batch, label_lengths)

    # optimizer
    train_step, global_step = nn_optimizer(
            loss, l2_scale, init_lr, dec_rate_lr, dec_rate_lr)

    summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(
                os.path.join(log_dir, "train"))
    validation_writer = tf.summary.FileWriter(
                os.path.join(log_dir, "validation"), tf.get_default_graph())


    # arguments to sess.run()
    train_list_to_run = [summaries, train_step, global_step, loss, acc_fgreedy]
    train_feed_dict = {dropout_keep_prob: dropout_keep_prob_const,
                       batch_norm_train_mode: True}
    val_list_to_run = [summaries, loss, acc_beam, edist, predicts, scores, global_step]
    val_feed_dict = {batch_norm_train_mode: False}
    
    return (train_list_to_run, train_feed_dict), (val_list_to_run, val_feed_dict), data_batch, label_batch



def get_spectrograms(data_batch, sampling_rate, frame_size=480.0,
                         frame_stride=160.0, num_mel_spec_bins=40.0):
    # input data_batch  -> (batch_size, sample_length)
    with tf.name_scope('audio_to_spec_conversion'):
        # get magnitude spectrogram via the short-term Fourier transform
        # (batch_size, num_frames, num_spectrogram_bins)
        mag_spectrogram = tf.abs(tf.contrib.signal.stft(
                                data_batch, frame_length=frame_size, frame_step=frame_stride,
                                fft_length=frame_size))
        num_mag_spec_bins = 1 + (frame_size // 2)

        # warp the linear scale to mel scale
        # [num_mag_spec_bins, num_mel_spec_bins]
        mel_weights = tf.contrib.signal.linear_to_mel_weight_matrix(
                num_mel_spec_bins, num_mag_spec_bins, sampling_rate,
                lower_edge_hertz=20.0, upper_edge_hertz=4000.0)

        # convert the magnitude spectrogram to mel spectrogram 
        # (batch_size, num_frames, num_mel_spec_bins)
        mel_spectrogram = tf.tensordot(mag_spectrogram , mel_weights, 1)
        mel_spectrogram.set_shape([mag_spectrogram .shape[0], 
                                   mag_spectrogram .shape[1], 
                                   num_mel_spec_bins])
                                   
        # FIX BELOW - whether to use log psectrogram or ordinary mel spectrogram                           
        scale_log = mel_spectrogram
        '''                           
        v_max = tf.reduce_max(mel_spectrogram, axis=[1, 2], keepdims=True)
        v_min = tf.reduce_min(mel_spectrogram, axis=[1, 2], keepdims=True)
        is_zero = tf.cast(tf.equal(v_max - v_min, 0), tf.float32)
        scale_mel = (mel_spectrogram - v_min) / (v_max - v_min + is_zero)

        epsilon = 0.001
        log_spectro = tf.log(scale_mel + epsilon)
        v_min = np.log(epsilon)
        v_max = np.log(epsilon + 1)
        
        scale_log = (log_spectro - v_min) / (v_max - v_min)
        '''
    # (batch_size, num_frames, num_mel_spec_bins)
    return scale_log

'''
def create_inference_graph(FLAGS, num_char_classes, max_encoding_length):
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

        label_lengths = tf.fill([tf.shape(spectrogram)[0]], max_encoding_length)
        logits_transposed = tf.transpose(logits, perm=[1, 0, 2])
        pred, score_cmp = tf.nn.ctc_beam_search_decoder(
                logits_transposed, label_lengths, top_paths=2)
        pred_cmp = [tf.cast(tf.sparse.to_dense(p), tf.int32)
                                            for p in pred]
        predicts = pred_cmp[0]
        scores = score_cmp[:, 0] - score_cmp[:, 1]
'''       
        















