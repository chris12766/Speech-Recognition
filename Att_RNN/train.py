import os
import glob
import tensorflow as tf
from att_RNN_model import *
from data_generator import DataGenerator
import sys
import scipy.io.wavfile
import multiprocessing



model_input_type = 1

# Training params
num_epochs = 100
init_lr = 0.001
lr_decay_steps = 25000
lr_decay_rate = 0.4


saves_dir = os.path.join(main_dir, "att_speech_project_saves_%d" % model_input_type)
if not os.path.isdir(saves_dir):
    os.mkdir(saves_dir)
log_dir = os.path.join(saves_dir, "att_logs_%d" % model_input_type)
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)
ckpt_dir = os.path.join(saves_dir, "att_ckpts_%d" % model_input_type)
if not os.path.isdir(ckpt_dir):
    os.mkdir(ckpt_dir)
    
    
def train_and_val():    
    # Data input pipeline
    data_gen = DataGenerator(batch_size, data_dir, model_input_type)
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
    train_args, val_args, x, y = create_train_graph(data_gen._num_classes,
                                                    data_gen._num_frames, data_gen._num_spec_bins, init_lr, lr_decay_steps, lr_decay_rate)

    # create savers
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    train_writer = tf.summary.FileWriter(os.path.join(log_dir, "train"), sess.graph)
    valid_writer = tf.summary.FileWriter(os.path.join(log_dir, "val"), sess.graph)

    # Load previous model version
    curr_step = 1
    best_val_accuracy = 0.0
    acc_summary = tf.Summary()
    model_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
    if model_checkpoint:
        print("Restoring from", model_checkpoint)
        saver.restore(sess=sess, save_path=model_checkpoint)
        filename_parts = model_checkpoint.split("-")
        curr_step += int(filename_parts[-1])
        best_val_accuracy = float(filename_parts[0].split("_")[-1])
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
        while True:
            try:
                data_batch, label_batch = sess.run(next_batch_train)
                
                feed_dict = train_args[1]
                feed_dict[x] = data_batch
                feed_dict[y] = label_batch
                
                summary, global_step, loss, _, learn_rate, global_norm = sess.run(train_args[0],
                                                                                                feed_dict=feed_dict)
                train_writer.add_summary(summary, curr_step) 
                
                print('curr_step #%d, epoch #%d' %(curr_step, epoch))
                print("Training stats: loss = %.4f" %loss)
                print("Learning rate:", learn_rate)
                print("Global norm:", global_norm)
                
                if curr_step % 100 == 0 or curr_step == 1 or epoch == num_epochs:
                    # Run validation every 100 steps
                    accuracy = validate(curr_step, epoch, x, y, sess, valid_writer, val_args, next_batch_val,
                                        len(labels_lists[1]), data_gen, val_iter_init_op, val_data_batch_plh, 
                                        val_label_batch_plh, data_lists, labels_lists, acc_summary)
                    if best_val_accuracy < accuracy:
                        best_val_accuracy = accuracy
                        checkpoint_prefix = os.path.join(ckpt_dir, "speech_input_%d_acc_%.5f" % (model_input_type, best_val_accuracy))
                        print("Saving in", checkpoint_prefix)
                        saver.save(sess, checkpoint_prefix, global_step=curr_step)
                
                curr_step += 1
            except tf.errors.OutOfRangeError:
                print()
                break
  
    sess.close()


def validate(curr_step, epoch, x, y, sess, valid_writer, val_args, next_batch_val, 
             val_dataset_size, data_gen, val_iter_init_op, val_data_batch_plh, 
             val_label_batch_plh, data_lists, labels_lists, acc_summary):
    loss_sum = 0
    confidence_sum = 0
    num_correct_preds = 0
    
    sess.run(val_iter_init_op, feed_dict={val_data_batch_plh: data_lists[1],
                                          val_label_batch_plh: labels_lists[1]})
    
    print("Validating...")
    while True:
        try:
            data_batch, label_batch = sess.run(next_batch_val)
            
            feed_dict = val_args[1]
            feed_dict[x] = data_batch
            feed_dict[y] = label_batch
            
            summary, global_step, loss, pred_values, pred_indices = sess.run(val_args[0],
                                                                             feed_dict=val_args[1])
            valid_writer.add_summary(summary, curr_step)

            
            loss_sum += loss
            confidence_sum += pred_values.sum()

            for i in range(label_batch.shape[0]):
                if label_batch[i] == pred_values[i]:
                    num_correct_preds += 1
        except tf.errors.OutOfRangeError:
            break
    
    
    # record accuracy of the model trained so far
    accuracy = (num_correct_preds / val_dataset_size)
    acc_summary.value.add(tag="accuracy", simple_value=accuracy)
    valid_writer.add_summary(acc_summary, curr_step)
    
    # calculate statistics
    print("Validation stats for step #%d epoch #%d:" % (curr_step, epoch)) 
    print("accuracy = %.5f" % accuracy)
    print("avg_loss = %.4f" % (loss_sum / val_dataset_size))
    print("confidence = %.3f" %(confidence_sum / val_dataset_size))

    return accuracy



train_and_val()    




