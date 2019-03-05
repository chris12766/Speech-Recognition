import numpy as np
import tensorflow as tf
from tensorflow.python.ops import io_ops
import os
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

flags = tf.app.flags
FLAGS = flags.FLAGS


# ADD HELP ARGUMENTS!!!!

# Additional parameters
flags.DEFINE_string("saves_dir", "D:\\speech_project_saves", "")
if not os.path.isdir(FLAGS.saves_dir):
    os.mkdir(FLAGS.saves_dir)
flags.DEFINE_string("result_dir", os.path.join(FLAGS.saves_dir, "run"), "")
if not os.path.isdir(FLAGS.result_dir):
    os.mkdir(FLAGS.result_dir)
flags.DEFINE_string("log_dir", os.path.join(FLAGS.result_dir, "logs"), "")
if not os.path.isdir(FLAGS.log_dir):
    os.mkdir(FLAGS.log_dir)
flags.DEFINE_string("ckpt_dir", os.path.join(FLAGS.result_dir, "ckpts"), "")
if not os.path.isdir(FLAGS.ckpt_dir):
    os.mkdir(FLAGS.ckpt_dir)
flags.DEFINE_float("bg_nsr", 0.5, "")
flags.DEFINE_float("bg_noise_prob", 0.75, "")
flags.DEFINE_float("unknown_pct", 1/6, "")
flags.DEFINE_integer("batch_size", 128, "")
flags.DEFINE_integer("num_epochs", 1, "")
flags.DEFINE_float("init_valida_epoch", 1, "")
flags.DEFINE_integer("num_submissions", 5, "")
# num_batches/dec_rate_lr = 2.3
flags.DEFINE_integer("lr_dec_step", 3800, "")
flags.DEFINE_integer("num_batches", 9000, "")
flags.DEFINE_string("silence_class", "9", "")
flags.DEFINE_integer("silence_len", "16000", "")

# Train parameters
flags.DEFINE_integer("sampling_rate", 16000, "")
flags.DEFINE_float("frame_size_ms", 30.0, "")
flags.DEFINE_float("frame_stride_ms", 10.0, "")
# fg_interp_factor = target_duration_ms/(target_duration_ms-pad_ms, "")
flags.DEFINE_float("pad_ms", 140, "")
flags.DEFINE_integer("target_duration_ms", 1140, "")
flags.DEFINE_integer("num_mel_bins", 46, "")
flags.DEFINE_string("file_words", "map_words.txt", "")
flags.DEFINE_string("file_chars", "map_chars.txt", "")
# the top num_key_words in map_words.txt are key words
flags.DEFINE_integer("num_keywords", 10, "")
flags.DEFINE_float("init_lr", 0.0002, "")
flags.DEFINE_float("dec_rate_lr", 0.3, "")
flags.DEFINE_float("keep_prob", 0.5, "")
flags.DEFINE_float("l2_scale", 0, "")

def layer_convbr(net, conv1_kernel, bn_train, pad='SAME', relu=True):
    """Network layer containing conv, batch_norm and relu.
    """
    weights = tf.Variable(
            tf.truncated_normal(conv1_kernel, stddev=0.01), name='kernel')
    net = tf.nn.conv2d(net, weights, [1, 1, 1, 1], pad)

    net = tf.layers.batch_normalization(net, training=bn_train)
    if relu:
        net = tf.nn.relu(net)
    return net


def layer_residual(net, conv_kernel, bn_train):
    """Residual layer containing two convs (with batch_norm) and relu.
    """
    org = net
    num_channels = net.shape[-1].value
    assert (num_channels == conv_kernel[-1]), "in_channels == out_channels!"

    net = layer_convbr(net, conv_kernel, bn_train)
    net = layer_convbr(net, conv_kernel, bn_train, relu=False)

    net = net + org
    net = tf.nn.relu(net)
    return net


def layer_convres_maxpool(net, conv_kernel, res_kernel, pool_kernel, bn_train):
    """Network layer containing conv, residual and relu.
    """
    net = layer_convbr(net, conv_kernel, bn_train, pad='VALID')
    net = layer_residual(net, res_kernel, bn_train)
    net = layer_residual(net, res_kernel, bn_train)
    tf.logging.debug('resnet_output.shape = %s', net.shape)

    net = tf.nn.max_pool(net, pool_kernel, pool_kernel, 'SAME')
    tf.logging.debug('maxpool_out.shape = %s', net.shape)
    return net


def layer_lstm(net, rnn_seq_length, rnn_num_hid, keep_prob, bn_train):
    """Backward LSTM layer.
    """
    # calculate depth, it is possible rnn_seq_length != net.shape[1]
    depth = np.prod(net.shape.as_list()[1:]) // rnn_seq_length
    net = tf.reshape(net, [-1, rnn_seq_length, depth])
    tf.logging.debug('lstm_input.shape = %s', net.shape)

    net = tf.reverse(net, axis=[1])
    fw_cell = tf.nn.rnn_cell.LSTMCell(rnn_num_hid, use_peepholes=True)
    fw_drop_cell = tf.nn.rnn_cell.DropoutWrapper(
            fw_cell, input_keep_prob=keep_prob, state_keep_prob=keep_prob,
            variational_recurrent=True, dtype=tf.float32, input_size=depth)

    output, _ = tf.nn.dynamic_rnn(fw_drop_cell, net, dtype=tf.float32,
                                                                time_major=False, scope='rnn')
    out_reverse = tf.reverse(output, axis=[1])
    net = tf.layers.batch_normalization(out_reverse, training=bn_train)
    return net


def layer_fcbr(net, num_classes, bn_train, relu=True):
    """Network layer containing feed-forward, batch_norm and leaky_relu.
    """
    num_units = net.shape[-1].value
    weights = tf.Variable(
            tf.truncated_normal([num_units, num_classes], stddev=0.01),
            name='kernel')
    net = tf.matmul(net, weights)

    net = tf.layers.batch_normalization(net, training=bn_train)
    if relu:
        net = tf.nn.leaky_relu(net, alpha=0.75)
    return net


def nn_conv_lstm(nn_input, num_classes, keep_prob, bn_train):
    """Builds the main graph.
        nn_input.shape:[batch_size, num_time_frame, num_mel_bins]
    """
    tf.logging.info('nn_input.shape = %s, is_training=%s',
                                    nn_input.shape, keep_prob != 1)
    dbg_layers = []
    dbg_embeddings = []
    dbg_layers.append(tf.expand_dims(nn_input, -1))

    with tf.name_scope('bn'):
        net = tf.layers.batch_normalization(nn_input, training=bn_train)
        net = tf.expand_dims(net, -1)

    with tf.name_scope('cnn'):
        # cnn_layer 1: conv, res, res, max_pool
        layer1_channel = 64
        conv1_kernel = [5, 3, 1, layer1_channel]
        res1_kernel = [3, 3, layer1_channel, layer1_channel]
        pool_kernel = [1, 2, 2, 1]
        net = layer_convres_maxpool(net, conv1_kernel, res1_kernel,
                                                                pool_kernel, bn_train)
        dbg_layers.append(net)

        # cnn_layer 2: conv, res, res, max_pool
        layer2_channel = 128
        conv2_kernel = [3, 3, layer1_channel, layer2_channel]
        res2_kernel = [3, 3, layer2_channel, layer2_channel]
        pool_kernel = [1, 2, 2, 1]
        net = layer_convres_maxpool(net, conv2_kernel, res2_kernel,
                                                                pool_kernel, bn_train)
        dbg_layers.append(net)

        # cnn_layer 3: conv, res, res, max_pool
        layer3_channel = 256
        conv3_kernel = [3, 3, layer2_channel, layer3_channel]
        res3_kernel = [3, 3, layer3_channel, layer3_channel]
        pool_kernel = [1, 2, 2, 1]
        net = layer_convres_maxpool(net, conv3_kernel, res3_kernel,
                                                                pool_kernel, bn_train)
        dbg_layers.append(net)
        tf.logging.info('cnn_out.shape = %s', net.shape)

    with tf.name_scope('rnn'):
        rnn_seq_length = net.shape[1].value
        rnn_num_hid = 768
        net = layer_lstm(net, rnn_seq_length, rnn_num_hid, keep_prob, bn_train)
        dbg_layers.append(net)
        dbg_embeddings.append(net)
    tf.logging.info('lstm_output.shape = %s', net.shape)

    with tf.name_scope('fc'):
        net = tf.reshape(net, [-1, net.shape[-1].value])
        tf.logging.debug('fc_input.shape = %s', net.shape)
        # fc_layer 1
        fc_num_hid = 160
        if keep_prob != 1:
            net = tf.nn.dropout(net, keep_prob)
        net = layer_fcbr(net, fc_num_hid, bn_train)
        dbg_embeddings.append(tf.reshape(net, [-1, rnn_seq_length, fc_num_hid]))
        tf.logging.debug('fc_output.shape = %s', net.shape)

        # fc_layer 2
        if keep_prob != 1:
            net = tf.nn.dropout(net, keep_prob)
        net = layer_fcbr(net, num_classes, bn_train, relu=False)
        tf.logging.debug('fc_output.shape = %s', net.shape)

        logits = tf.reshape(net, [-1, rnn_seq_length, num_classes])
        dbg_embeddings.append(logits)
        tf.logging.info('fc_logits.shape = %s', logits.shape)

    return logits, dbg_layers, dbg_embeddings


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

def create_train_graph(num_classes, max_label_length):
    audio_length = int(FLAGS.sampling_rate * FLAGS.target_duration_ms / 1000)
    frame_size = int(FLAGS.sampling_rate * FLAGS.frame_size_ms / 1000)
    frame_stride = int(FLAGS.sampling_rate * FLAGS.frame_stride_ms / 1000)
    
    

    audio = tf.placeholder(tf.float32, [None, audio_length],name="audio")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    bn_train = tf.placeholder(tf.bool, name='bn_train')
    
    # convert audio to spectrogram
    nn_input = audio_to_spectrogram(
            audio, FLAGS.sampling_rate, frame_size, frame_stride, FLAGS.num_mel_bins)

    # network base
    logits, _, _ = nn_conv_lstm(nn_input, num_classes, keep_prob, bn_train)

    # ctc_cost function
    label_lengths = tf.fill([tf.shape(nn_input)[0]], max_label_length)
    labels = tf.placeholder(tf.int32, [None, max_label_length], name="label")
    loss, acc_fgreedy, acc_beam, edist, \
            predicts, scores, raw_preds, \
            max_probs = nn_cost_ctc(logits, labels, label_lengths)

    # optimizer
    train_step, global_step = nn_optimizer(
            loss, FLAGS.l2_scale, FLAGS.init_lr, FLAGS.dec_rate_lr, FLAGS.dec_rate_lr)

    summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(
                os.path.join(FLAGS.log_dir, "train"))
    validation_writer = tf.summary.FileWriter(
                os.path.join(FLAGS.log_dir, "validation"), tf.get_default_graph())


    # arguments to sess.run()
    train_list_to_run = [summaries, train_step, global_step, loss, acc_fgreedy]
    train_feed_dict = {keep_prob: FLAGS.keep_prob,
                       bn_train: True}
    val_list_to_run = [summaries, loss, acc_beam, edist, predicts, scores, global_step]
    val_feed_dict = {keep_prob: 1,
                     bn_train: False}
    
    return (train_list_to_run, train_feed_dict), (val_list_to_run, val_feed_dict), audio, labels

def audio_to_spectrogram(audio, sampling_rate, frame_size=480.0,
                                                 frame_stride=160.0, num_mel_bins=40.0, debug=False):
    """Builds a TensorFlow graph to convert audio to spectrogram.
        audio.shape: [batch_size, sample_length]
        magnit_spectro.shape:[batch_size, num_time_frame, num_spectrogram_bins]
        scale_log.shape: [batch_size, num_time_frame, num_mel_bins]
    """

    with tf.name_scope('audio_to_spectrogram'):
        magnit_spectro = tf.abs(tf.contrib.signal.stft(
                audio, frame_length=frame_size, frame_step=frame_stride,
                fft_length=frame_size))
        num_spectro_bins = magnit_spectro.shape[-1].value

        mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
                num_mel_bins, num_spectro_bins, sampling_rate,
                lower_edge_hertz=20.0, upper_edge_hertz=4000.0)

        mel_spectro = tf.tensordot(
                magnit_spectro, mel_weight_matrix, 1)
        mel_spectro.set_shape(magnit_spectro.shape[:-1].concatenate(
                mel_weight_matrix.shape[-1:]))
        v_max = tf.reduce_max(mel_spectro, axis=[1, 2], keepdims=True)
        v_min = tf.reduce_min(mel_spectro, axis=[1, 2], keepdims=True)
        is_zero = tf.to_float(tf.equal(v_max - v_min, 0))
        scale_mel = (mel_spectro - v_min) / (v_max - v_min + is_zero)

        epsilon = 0.001
        log_spectro = tf.log(scale_mel + epsilon)
        v_min = np.log(epsilon)
        v_max = np.log(epsilon + 1)
        scale_log = (log_spectro - v_min) / (v_max - v_min)
    if debug:
        return magnit_spectro, mel_spectro, scale_mel, scale_log
    else:
        return scale_log

'''
def create_inference_graph(FLAGS, num_classes, max_label_length):
        audio_length = int(FLAGS.sampling_rate * FLAGS.target_duration_ms / 1000)
        frame_size = int(FLAGS.sampling_rate * FLAGS.frame_size_ms / 1000)
        frame_stride = int(FLAGS.sampling_rate * FLAGS.frame_stride_ms / 1000)
        FLAGS.num_mel_bins = FLAGS.FLAGS.num_mel_bins
        tf.logging.info('audios_inference parameters: %s',
                                        [audio_length, frame_size, frame_stride, FLAGS.num_mel_bins])

        audio = tf.placeholder(tf.float32, [None, audio_length],
                                                                 name='audio')
        nn_input = audio_to_spectrogram(
                audio, FLAGS.sampling_rate, frame_size, frame_stride, FLAGS.num_mel_bins)

        bn_train = tf.placeholder(tf.bool, name='bn_train')
        logits, dbg_layers, dbg_embeddings = nn_conv_lstm(
                nn_input, num_classes, keep_prob=1, bn_train=bn_train)
        prob = tf.nn.softmax(logits)
        chars = tf.argmax(prob, axis=2)

        label_lengths = tf.fill([tf.shape(nn_input)[0]], max_label_length)
        logits_transposed = tf.transpose(logits, perm=[1, 0, 2])
        pred, score_cmp = tf.nn.ctc_beam_search_decoder(
                logits_transposed, label_lengths, top_paths=2)
        pred_cmp = [tf.cast(tf.sparse.to_dense(p), tf.int32)
                                            for p in pred]
        predicts = pred_cmp[0]
        scores = score_cmp[:, 0] - score_cmp[:, 1]
'''       
        















