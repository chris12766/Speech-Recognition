import os
import glob
import tensorflow as tf
from conv_lstm_ctc_net import *
from data_generator import DataGenerator
from word_dictionary import WordDict
import sys
import scipy.io.wavfile
import multiprocessing

# audio process pool, multi-processes to speed up audio augmentation.
# non-deterministic.
#_audio_process_pool = None
# enable multiprocessing debugging
# multiprocessing.log_to_stderr().setLevel(multiprocessing.SUBDEBUG)

#_bg_audios = []

def train_and_eval():
    # Set up word_dictionary and character_dictionary.
    word_dict = WordDict(FLAGS.file_words, FLAGS.file_chars, FLAGS.num_keywords, FLAGS.silence_class)
    num_words, num_char_classes = word_dict.num_classes
    max_label_length = word_dict.max_label_length
    
    
    # Data input pipeline
    data_gen = DataGenerator(word_dict, FLAGS.batch_size, FLAGS.unknown_pct,
                             FLAGS.silence_class, FLAGS.silence_len, FLAGS.sampling_rate, FLAGS.target_duration_ms)
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
    train_args, val_args, x, y = create_train_graph(num_char_classes, max_label_length)

    # create savers
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100000)
    train_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, "train"), sess.graph)
    valid_writer = tf.summary.FileWriter(os.path.join(FLAGS.log_dir, "validation"), sess.graph)

    # Load previous model version
    step = 1
    model_checkpoint = tf.train.latest_checkpoint(FLAGS.ckpt_dir)
    if model_checkpoint:
        print("Restoring from", model_checkpoint)
        saver.restore(sess=sess, save_path=model_checkpoint)
        step += int(model_checkpoint.split("-")[-1])
    else:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

    for epoch in range(1, FLAGS.num_epochs + 1):
        print("Epoch number: %d" %epoch)
        data_lists, labels_lists = data_gen._get_data_lists()
        placeholders = data_gen._get_dataset_placeholders()
        train_data_batch_plh, train_label_batch_plh = placeholders[0]
        val_data_batch_plh, val_label_batch_plh = placeholders[1]
            
        print(data_lists[0].shape)
        print(data_lists[0])
        print(labels_lists[0].shape)
        print(labels_lists[0])
        print(train_data_batch_plh)
        print(train_label_batch_plh)
        print()
            
        sess.run(train_iter_init_op, feed_dict={train_data_batch_plh: data_lists[0],
                                                train_label_batch_plh: labels_lists[0]})
        sess.run(val_iter_init_op, feed_dict={val_data_batch_plh: data_lists[1],
                                              val_label_batch_plh: labels_lists[1]})
        while True:
            try:
                # get data   
                # (128, )         (128, 4)
                wav_paths_batch, label_batch = sess.run(next_batch_train)
                #decoded_audio = decode_wav_batch(wav_paths_batch)
                # (128, 18240)
                decoded_audio = np.array(list(map(decode_wav, wav_paths_batch)))
                
                feed_dict = train_args[1]
                feed_dict[x] = decoded_audio
                feed_dict[y] = label_batch
                
                summary, _, global_step, loss, acc_fgreedy = sess.run(train_args[0],feed_dict=feed_dict)
                train_writer.add_summary(summary, step) 
                
                print('Step #%d, epoch #%d' %(step, epoch))
                print("Training stats: acc_fgreedy = %.2f, loss = %.4f" %(acc_fgreedy * 100, loss))
                
                
                
                if step > 5 and step <= 10:
                    word_dict.word_distro(label_batch)

                
                #  Run validation and save ckpt.
                evaluate(step, epoch, x, y, sess, valid_writer, word_dict, val_args, next_batch_val, len(labels_lists[1]))
                step += 1
                
                sys.exit()
            except tf.errors.OutOfRangeError:
                print("Epoch ended")
                break
        
        # save after each epoch
        print("Saving in", ckpt_path)
        saver.save(sess, ckpt_path, global_step=step)
  
    sess.close()

'''
def init_audio_process(bg_audios=None):
    
    #Initialization function for creating audio processes.
    #- Disables SIGINT in child processes and leave the main process to turn off
    #the daemon child processes.
    #- Updates _bg_audios, so that AudioAugmentor in child processes can use
    #bg audio directly without getting it from the parent process repeatedly.
    #- Resets random seed, so that different child processes have different
    #random seeds, not the same one as the parent process.
    
    def sig_handler(*unused):
        # doing_nothing sig_handler
        return None
    signal.signal(signal.SIGINT, sig_handler)

    if bg_audios:
        global _bg_audios
        _bg_audios = bg_audios

        identity = multiprocessing.current_process()._identity
        np.random.seed(sum(identity) ** 2)

    print('audio_pid=%s, bg_audios=%d' % (os.getpid(), len(_bg_audios)))

def get_audio_pool(initargs):
    # Returns the audio process pool. Creates the pool if it does not exist.
    
    global _audio_process_pool
    if _audio_process_pool is None:
        _audio_process_pool = multiprocessing.Pool(initializer=init_audio_process,initargs=initargs)
    return _audio_process_pool

def decode_wav_batch(wav_paths_batch):
    initargs = [] # clean audio or use [bg_audios] for noisy
    
    audio_process_pool = get_audio_pool(initargs)
    audios_list = audio_process_pool.map(decode_wav, wav_paths_batch)
    audios = np.vstack(audios_list)
    
    return audios

'''

# if too slow, do it with tf map
def decode_wav(wav_path):
    INT15_SCALE = np.power(2, 15)
    target_length = int(FLAGS.sampling_rate * FLAGS.target_duration_ms / 1000)
    
    
    _, decoded_audio = scipy.io.wavfile.read(wav_path)
    decoded_audio = decoded_audio.astype(np.float32, copy=False)
    # keep 0 not shifted. Not: (audio + 0.5) * 2 / (INT15_SCALE * 2 - 1)
    decoded_audio /= INT15_SCALE
    curr_audio_length = len(decoded_audio)

    # fix audio length by cutting or appending equally from both ends
    start_index = abs(target_length - curr_audio_length) // 2
    if curr_audio_length < target_length:
        audio_reformatted = np.zeros(target_length, dtype=np.float32)
        audio_reformatted[start_index : start_index + curr_audio_length] = decoded_audio
    elif curr_audio_length > target_length:
        audio_reformatted = decoded_audio[start_index : start_index + target_length]

    return audio_reformatted



def evaluate(step, epoch, x, y, sess, valid_writer, word_dict, val_args, next_batch_val, val_dataset_size):
    avg_acc_beam = 0
    avg_loss = 0
    sum_score = 0
    wrong_submits = 0
    sum_edist = 0
    
    while True:
        try:
            wav_paths_batch, label_batch = sess.run(next_batch_val)
            decoded_audio = np.array(list(map(decode_wav, wav_paths_batch)))
            
            feed_dict = val_args[1]
            feed_dict[x] = decoded_audio
            feed_dict[y] = label_batch
            
            summary, loss, acc_beam, edist, predictions, scores, global_step = sess.run(
                                                                                val_args[0],
                                                                                feed_dict=val_args[1])
            valid_writer.add_summary(summary, global_step)

            
            scale = FLAGS.batch_size * 100 / val_dataset_size
            avg_acc_beam += acc_beam * scale
            avg_loss += loss * scale
            sum_score += scores.sum()
            sum_edist += edist.sum() / val_dataset_size

            for i in range(FLAGS.batch_size):
                true_submit = word_dict.indices_to_submit(label_batch[i])
                pred_submit = word_dict.indices_to_submit(predictions[i])
                wrong_submits += int(pred_submit != true_submit)
                
                
            break
        except tf.errors.OutOfRangeError:
            break
    

    acc_submit = (1 - wrong_submits / val_dataset_size) * 100
    print('Step #%d, epoch #%d' %(step, epoch))
    print("Validation stats: acc_beam = %.2f, acc_submit = %.2f" %(avg_acc_beam, acc_submit))
    print("loss = %.4f, sum_edist = %.4f, confidence = %.3f" %(avg_loss/100,sum_edist,sum_score/val_dataset_size))





train_and_eval()    




