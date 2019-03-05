import os
import numpy as np
import tensorflow as tf
import re
from tensorflow.data import Dataset
import collections
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile
import scipy.io.wavfile

train_data_dir = "D:/speech_data/train/"
val_data_dir = "D:/speech_data/val/"
test_data_dir = "D:/speech_data/test/"


class DataGenerator(object):

    def __init__(self, word_dict, batch_size, unknown_pct, silence_class, silence_len, sampling_rate, target_duration_ms, buffer_size=1000):
        self.silence_class = silence_class
        self.data_size = 0
        self.target_duration_ms = target_duration_ms
        self.sampling_rate = sampling_rate
        self.silence_len = silence_len
        self.unknown_pct = unknown_pct
        self.batch_size = batch_size
        self.word_dict = word_dict
        # retrieve the data from the text file
        self._process_files([train_data_dir, val_data_dir])
        self.datasets = []
        self.placeholders = []

        for i in range(len(self.data_lists)):
            labels = self.labels_lists[i]
            data = self.data_lists[i]
            # create dataset
            data_placeholder = tf.placeholder(data.dtype, data.shape, name=("data_placeholder_%d" % i))
            labels_placeholder = tf.placeholder(labels.dtype, labels.shape, name=("labels_placeholder_%d" % i))
            self.placeholders.append((data_placeholder, labels_placeholder))
            dataset = Dataset.from_tensor_slices((data_placeholder, labels_placeholder))
            # parse images
            dataset = dataset.batch(batch_size, drop_remainder=False)
            self.datasets.append(dataset)
        
        
    def _get_datasets(self):
    	return self.datasets
        
    def _get_dataset_placeholders(self):
        return self.placeholders
        
    def _get_data_lists(self):
        return self.data_lists, self.labels_lists


    '''
    def _num_total_batches(self):
        if self.data_size == 0:
            for list in self.data_lists:
                self.data_size += len(list)
    '''


    def _process_files(self, dir_list):
        self.data_lists = []
        self.labels_lists = []
    
        for dir in dir_list:
            known, unknown = self.load_data_from_dir(train_data_dir)
            # shuffle unknown samples and take only a percentage of them
            np.random.shuffle(unknown)
            unknown = unknown[:int(len(known) * self.unknown_pct)]
            
            # known is now the whole dataset
            known.extend(unknown)
            np.random.shuffle(known)
            
            data_list, label_list = zip(*known)
            
            self.data_lists.append(np.array(data_list))
            self.labels_lists.append(np.array(label_list))
        
    def load_data_from_dir(self, data_dir):
        known = []
        unknown = []
        missing_words = set()
        
        # to find the most common known word
        known_word_occur = collections.Counter()

        key_words = self.word_dict.key_words
        # go over all WAV files in directory and sub-directories
        for wav in gfile.Glob(os.path.join(data_dir, '*', '*nohash*.wav')):
            if not wav.endswith(".wav"):
                continue
            curr_word = wav.split("\\")[-2].lower()

            # get labels
            indices = self.word_dict.word_to_indices(curr_word)
            if indices:
                if curr_word in key_words:
                    known.append([wav, indices])
                    known_word_occur[curr_word] += 1
                else:
                    unknown.append([wav, indices])
            else:
                missing_words.add(curr_word)
        print("Words missing from word_map.txt:", missing_words)

        
        silence_dir = os.path.join(data_dir, self.silence_class)
        if not os.path.exists(silence_dir):
            os.makedirs(silence_dir)
        
        # make silence waves (all zeros) and add samples equal to the most common word
        silence_file_path = os.path.join(silence_dir, "silence_0.wav").replace("/", "\\")
        self.encode_audio(np.zeros([self.silence_len]), silence_file_path)
        
        silence_indices = self.word_dict.word_to_indices(self.silence_class)
        silence_size = known_word_occur.most_common(1)[0][1]
        known.extend([[silence_file_path, silence_indices]] * silence_size)

        return known, unknown
        
    def encode_audio(self, audio, wav_file, sampling_rate=16000):
        '''
        Encodes to audio wave file.
        audio.shape:[target_length]
        '''
        INT15_SCALE = np.power(2, 15)
        audio = audio * INT15_SCALE
        audio = np.clip(audio, -INT15_SCALE, INT15_SCALE - 1)
        audio = audio.reshape([-1]).astype(np.int16, copy=False)
        
        print("saving encoded audio to %s"  %wav_file)
        scipy.io.wavfile.write(wav_file, sampling_rate, audio)
        
