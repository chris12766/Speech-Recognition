"""The main training script, to train a Conv-LSTM-CTC network on
    Google Speech Commands data set and run predictions on the test set.
"""

import os
import sys
import time
import argparse
import glob

import tensorflow as tf

import util_train
import util_data
import audio_dataset
import models

FLAGS = None
MODEL_FLAGS = None


def main(_):
    t_stamp, init_step = util_train.train_init(FLAGS.fold_idx, FLAGS.start_ckpt,
                                                                                         FLAGS.result_dir)

    tf.logging.debug('fold_idx=%d, MODEL_FLAGS=%s', FLAGS.fold_idx, MODEL_FLAGS)

    # Set up word_dictionary and character_dictionary.
    word_dict = util_data.WordDict(FLAGS.file_words, FLAGS.file_chars,
                                                                 FLAGS.num_key_words)
    num_words, num_char_classes = word_dict.num_classes
    max_label_length = word_dict.max_label_length
    tf.logging.debug('num_words=%d, num_chars=%d, max_label_length=%d',
                                     num_words, num_char_classes, max_label_length)

    audio_length = int(FLAGS.sampling_rate * FLAGS.target_duration_ms / 1000)
    audio_pad = int((FLAGS.pad_ms * FLAGS.sampling_rate) / 1000)
    batch_size = FLAGS.batch_size
    # Set up datasets for training and validation.
    dataset_train, dataset_valida = audio_dataset.load_datasets(
            FLAGS.data_url, FLAGS.train_data_dir, word_dict, FLAGS.unknown_pct,
            FLAGS.num_folds, FLAGS.fold_idx, audio_length, audio_pad,
            FLAGS.bg_noise_prob, FLAGS.bg_nsr)
    tf.logging.info('dataset size: training=%d, validation=%d, T/V=%.2f\n',
                                    dataset_train.size, dataset_valida.size,
                                    dataset_train.size / dataset_valida.size)

    # Start a TensorFlow session.
    sess = util_train.tf_session(FLAGS.gmem_dynamic)

    # Define model graph.
    dec_rate_lr = FLAGS.dec_rate_lr
    model_graph = models.AudioNN(FLAGS, num_char_classes, max_label_length,
                                                             dec_rate_lr, is_training=True,
                                                             log_dir=FLAGS.log_dir)
    tf.train.write_graph(sess.graph_def, FLAGS.result_dir,
                                             'audio_nn_graph_%s.pbtxt' % t_stamp)

    # samples used to monitor how training evolves.
    samples = [os.path.join(FLAGS.train_data_dir, 'right/0132a06d_nohash_0.wav'),
                         os.path.join(FLAGS.train_data_dir, 'right/10ace7eb_nohash_2.wav'),
                         os.path.join(FLAGS.train_data_dir, 'four/c9b653a0_nohash_3.wav')]

    # Set up validation evaluator and checkpoint recorder.
    evaluator = util_train.Evaluator(
            dataset_valida, 'validation', model_graph, sess, batch_size,
            word_dict, FLAGS.result_dir, t_stamp, audio_length, samples)
    ckpter = util_train.Ckpter(sess, FLAGS.ckpt_dir, 'ann', t_stamp,
                                                         max_ckpt=30, num_best=FLAGS.num_submissions)

    epoch_size = dataset_train.size
    num_steps = FLAGS.num_batches
    num_epoch = num_steps * batch_size / epoch_size
    num_unknown_epoch = (num_steps * batch_size * FLAGS.unknown_pct /
                                             dataset_train.list_size[-1])
    valida_interval = round(FLAGS.init_valida_epoch * epoch_size / batch_size)
    valida_interval = 0.5 ** (init_step // dec_rate_lr) * valida_interval
    steps_per_epoch = int(round(epoch_size / batch_size))
    tf.logging.info('num_epochs=[%.2f, %.2f], num_steps=%d, lr_decay_step=%d, '
                                    'steps_per_epoch=%d, initial_validation_steps=%d',
                                    num_epoch, num_unknown_epoch, num_steps, dec_rate_lr,
                                    steps_per_epoch, valida_interval)

    if FLAGS.start_ckpt:
        ckpter.restore(FLAGS.start_ckpt)
        tf.logging.debug('step from ckpt: %d', model_graph.eval_global_step(sess))
    else:
        tf.global_variables_initializer().run(session=sess)
        ckpter.save(init_step)

    if util_train.dbg_basic():
        sys.exit()

    evaluator.samples_eval(init_step)
    # Starting from step+1, to match graph.global_step.
    start_step = init_step + 1
    tf.logging.info('Training from step %d.\n', start_step)

    # Start training..
    prev = time.time()
    for step in range(start_step, num_steps + 1):
        perf_detail = step > 15 and step < 20
        t0 = perf_detail and time.time()
        data, labels, _ = dataset_train.next_batch(batch_size, batch_wrap=True,
                                                                                             shuffle=True)
        t1 = perf_detail and time.time()
        loss, acc_fgreedy = model_graph.eval_train(sess, data, labels,
                                                                                             FLAGS.dropout_keep_prob)
        t2 = perf_detail and time.time()
        if perf_detail:
            tf.logging.debug('data_augment=%.3fs, training=%.3fs', t1 - t0, t2 - t1)

        epoch_step = step * batch_size / epoch_size
        if step % 25 == 0:
            tf.logging.info('Step #%d, epoch #%.2f: acc_fgreedy = %.2f%%, '
                                            'loss = %.4f, time = %.3fs', step, epoch_step,
                                            acc_fgreedy * 100, loss, time.time() - prev)
            prev = time.time()

        if step > 5 and step <= 10:
            word_dict.word_distro(labels)

        # Run validation and save ckpt.
        if step % dec_rate_lr == 0:
            valida_interval = max(int(0.5 * valida_interval), 1)
        if step % valida_interval == 0 or step == 1:
            tf.logging.info('Step #%d, epoch #%.2f [validation]:', step, epoch_step)
            score = evaluator.run_evaluation(step)
            ckpter.record(step, score)

        if step < 1500 or step > num_steps - 1000:
            evaluator.samples_eval(step)
        elif step % valida_interval == 0:
            evaluator.samples_eval(step)

    # Run inference on test set.

    best_ckpts = ckpter.get_best()
    if not FLAGS.skip_test:
        for ckpt in best_ckpts:
            # training +validation data
            wav_list = glob.glob(os.path.join(FLAGS.train_data_dir,
                                                                                '*', '*nohash*.wav'))
            util_train.infer_testset(model_graph, sess, audio_length, word_dict,
                                                             wav_list, FLAGS.result_dir, batch_size,
                                                             ckpter.saver, ckpt, mode='train')
            # test data
            wav_list = glob.glob(os.path.join(FLAGS.test_data_dir, '*wav'))
            util_train.infer_testset(model_graph, sess, audio_length, word_dict,
                                                             wav_list, FLAGS.result_dir, batch_size,
                                                             ckpter.saver, ckpt)
            # validation stats
            evaluator.stats_validation(ckpt.split('-')[-1])

    sess.close()