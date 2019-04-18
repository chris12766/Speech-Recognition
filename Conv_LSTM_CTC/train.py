import os
import glob
import tensorflow as tf
from conv_lstm_ctc_net import *
from data_generator import DataGenerator
import sys
import scipy.io.wavfile
import multiprocessing


def train_and_eval():    
    # Data input pipeline
    data_gen = DataGenerator(batch_size, data_dir)
    datasets = data_gen._get_datasets()
    
    train_dataset = datasets[0]
    train_iterator = train_dataset.make_initializable_iterator()
    train_iter_init_op = train_iterator.initializer
    next_batch_train = train_iterator.get_next()
    
    val_dataset = datasets[1]
    val_iterator = val_dataset.make_initializable_iterator()
    val_iter_init_op = val_iterator.initializer
    next_batch_val = val_iterator.get_next()
  

    # Session start
    config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth = True), 
                            allow_soft_placement=True, 
                            log_device_placement=False)                      
    sess = tf.Session(config=config)
    

    # Create train graph
    train_args, val_args, x, y = create_train_graph(data_gen._num_char_classes, data_gen._label_encoding_length)

    # create savers
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100000)
    train_writer = tf.summary.FileWriter(os.path.join(log_dir, "train"), sess.graph)
    valid_writer = tf.summary.FileWriter(os.path.join(log_dir, "validation"), sess.graph)

    # Load previous model version
    curr_step = 1
    model_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
    if model_checkpoint:
        print("Restoring from", model_checkpoint)
        saver.restore(sess=sess, save_path=model_checkpoint)
        curr_step += int(model_checkpoint.split("-")[-1])
    else:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

    print("Start of training...")
    for epoch in range(1, num_epochs + 1):
        print("Epoch number: %d" %epoch)
        data_lists, labels_lists = data_gen._get_data_lists()
        placeholders = data_gen._get_dataset_placeholders()
        train_data_batch_plh, train_label_batch_plh = placeholders[0]
        val_data_batch_plh, val_label_batch_plh = placeholders[1]
            
        sess.run(train_iter_init_op, feed_dict={train_data_batch_plh: data_lists[0],
                                                train_label_batch_plh: labels_lists[0]})
        sess.run(val_iter_init_op, feed_dict={val_data_batch_plh: data_lists[1],
                                              val_label_batch_plh: labels_lists[1]})
        while True:
            try:
                # (128, )    (128, 4)
                data_batch, label_batch = sess.run(next_batch_train)
                
                feed_dict = train_args[1]
                feed_dict[x] = data_batch
                feed_dict[y] = label_batch
                
                
                summary, global_step, loss, acc_greedy, edit_dist_greedy, \
                        acc_beam, edit_dist_beam, scores, _ = sess.run(train_args[0],
                                                                       feed_dict=feed_dict)
                train_writer.add_summary(summary, curr_step) 
                
                print('curr_step #%d, epoch #%d' %(curr_step, epoch))
                print("Training stats: acc_greedy = %.2f, loss = %.4f" %(acc_greedy * 100, loss))
                
                
                #  Run validation and save ckpt.
                evaluate(curr_step, epoch, x, y, sess, valid_writer, val_args, next_batch_val, len(labels_lists[1]), data_gen)
                curr_step += 1
                
                
                
                
                
                
                break
                
                
                
                
                
            except tf.errors.OutOfRangeError:
                print("Epoch ended")
                break
        
        # save after each epoch
        print("Saving in", ckpt_dir)
        saver.save(sess, ckpt_dir, global_step=curr_step)
  
    sess.close()


def evaluate(curr_step, epoch, x, y, sess, valid_writer, val_args, next_batch_val, val_dataset_size, data_gen):
    avg_acc_beam = 0
    avg_loss = 0
    sum_score = 0
    wrong_submits = 0
    sum_edist = 0
    
    while True:
        try:
            data_batch, label_batch = sess.run(next_batch_val)
            
            feed_dict = val_args[1]
            feed_dict[x] = data_batch
            feed_dict[y] = label_batch
            
            summary, global_step, loss, acc_greedy, edit_dist_greedy, \
                        acc_beam, edit_dist_beam, scores, predictions = sess.run(val_args[0],
                                                                                 feed_dict=val_args[1])
            valid_writer.add_summary(summary, global_step)

            
            scale = batch_size * 100 / val_dataset_size
            avg_acc_beam += acc_beam * scale
            avg_loss += loss * scale
            sum_score += scores.sum()
            sum_edist += edit_dist_beam.sum() / val_dataset_size

            for i in range(batch_size):
                true_submit = data_gen._get_label_from_encoding(label_batch[i])
                pred_submit = data_gen._get_label_from_encoding(predictions[i])
                wrong_submits += int(pred_submit != true_submit)
                
            




            
            break
            
            
            
            
            
        except tf.errors.OutOfRangeError:
            break
    

    acc_submit = (1 - wrong_submits / val_dataset_size) * 100
    print('curr_step #%d, epoch #%d' %(curr_step, epoch))
    print("Validation stats: acc_beam = %.2f, acc_submit = %.2f" %(avg_acc_beam, acc_submit))
    print("loss = %.4f, sum_edist = %.4f, confidence = %.3f" %(avg_loss/100,sum_edist,sum_score/val_dataset_size))





train_and_eval()    




