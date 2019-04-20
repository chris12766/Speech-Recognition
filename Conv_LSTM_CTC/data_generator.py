import os
import numpy as np
import tensorflow as tf
import collections
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile
import scipy.io.wavfile
import concurrent.futures
import re



class DataGenerator(object):

    def __init__(self, batch_size, data_dir, input_type):
        self._train_data_dir = os.path.join(data_dir, "train")
        self._val_data_dir = os.path.join(data_dir, "val")
        self._test_data_dir = os.path.join(data_dir, "test")
        self._unknown_data_ratio = 1/6
        
        self._bad_paths = []
        bad_file_records = ["bad_samples.txt", "silences.txt", "bad_samples_2.txt"]
        for f in bad_file_records:
            with open(os.path.join(data_dir, f)) as fp:  
                line = fp.readline()
                while line:
                    self._bad_paths.append(line)
                    line = fp.readline()
        
        
        # Data params
        self._bg_nsr = 0.5
        self._bg_noise_prob = 0.75
        self._sampling_rate = 16000
        self._frame_size_ms = 30.0
        self._frame_stride_ms = 10.0
        
        self._padding_ms = 140
        self._audio_dur_in_ms = 1140
        self._num_spec_bins = 46
        self._num_frames = 112
        self._audio_length = int(self._sampling_rate * self._audio_dur_in_ms / 1000) # 18,240
        self._frame_size = int(self._sampling_rate * self._frame_size_ms / 1000)     # 480
        self._frame_stride = int(self._sampling_rate * self._frame_stride_ms / 1000) # 160
        
        
        
        self._silence_word = "silence"
        self._silence_alias = "9"
        self._unknown_label = "unknown"
        self._silence_length = 16000
        self._label_encoding_length = 4 
        
        # the first _num_known_words are known
        self._num_known_words = 10
        self._word_to_alias = {"yes" : "yes",
                                "no" : "nO",
                                "up" : "up",
                                "down" : "dUn",
                                "left" : "left",
                                "right" : "rIt",
                                "on" : "on",
                                "off" : "of",
                                "stop" : "sdop",
                                "go" : "gO",
                                "zero" : "sirO",
                                "one" : "wun",
                                "two" : "tw",
                                "three" : "srE",
                                "four" : "fw",
                                "five" : "fIv",
                                "six" : "sics",
                                "seven" : "sevn",
                                "eight" : "Et",
                                "nine" : "nIn",
                                "house" : "hUs",
                                "happy" : "hapi",
                                "bird" : "bwd",
                                "wow" : "wU",
                                "bed" : "bed",
                                "dog" : "dog",
                                "cat" : "cat",
                                "marvin" : "mavn",
                                "sheila" : "sEli",
                                self._silence_word : self._silence_alias}
        self._alias_to_word = {v: k for k, v in self._word_to_alias.items()}
        self._known_words = [*self._word_to_alias][:self._num_known_words]
        

        # used to classify
        self._alias_to_label = {}
        for alias, word in self._alias_to_word.items():
            if word in self._known_words or word == self._silence_word:
                self._alias_to_label[alias] = word

        # for encoding of aliases
        self._id_to_char = {0 : "",
                            1 : "U",
                            2 : "9",
                            3 : "m",
                            4 : "r",
                            5 : "o",
                            6 : "b",
                            7 : "I",
                            8 : "y",
                            9 : "d",
                            10 : "l",
                            11 : "h",
                            12 : "c",
                            13 : "e",
                            14 : "g",
                            15 : "n",
                            16 : "O",
                            17 : "f",
                            18 : "u",
                            19 : "p",
                            20 : "t",
                            21 : "w",
                            22 : "v",
                            23 : "E",
                            24 : "s",
                            25 : "i",
                            26 : "a",
                            27 : "_"}
        self._char_to_id = {v: k for k, v in self._id_to_char.items()}
        self._num_char_classes = len(self._id_to_char)
        
        
        # retrieve the data
        self._load_data_in_lists([self._train_data_dir, self._val_data_dir])
        self._datasets = []
        self._placeholders = []

        for i in range(len(self._data_lists)):
            labels = self._labels_lists[i]
            data = self._data_lists[i]
            # create datasets
            data_placeholder = tf.placeholder(data.dtype, data.shape, name=("data_placeholder_%d" % i))
            labels_placeholder = tf.placeholder(labels.dtype, labels.shape, name=("labels_placeholder_%d" % i))
            self._placeholders.append((data_placeholder, labels_placeholder))
            dataset = tf.data.Dataset.from_tensor_slices((data_placeholder, labels_placeholder))
            
            # use the whole dataset, model does not rely on equal sized batches
            dataset = dataset.batch(batch_size, drop_remainder=False)
            
            
            if input_type == 0:   # decoded wav (PCM)
                dataset = dataset.map(lambda x,y : (self._modify_PCM(x), y), num_parallel_calls=20)
            elif input_type == 1: # mag spectrogram
                dataset = dataset.map(lambda x,y : (self._convert_to_mag_specs(x), y), num_parallel_calls=20)
            elif input_type == 2: # log mag spectrogram
                dataset = dataset.map(lambda x,y : (self._convert_to_log_mag_specs(x), y), num_parallel_calls=20)
            elif input_type == 3: # mel spectrogram
                dataset = dataset.map(lambda x,y : (self._convert_to_mel_specs(x), y), num_parallel_calls=20)
            elif input_type == 4: # log mel spectrogram
                dataset = dataset.map(lambda x,y : (self._convert_to_log_mel_specs(x), y), num_parallel_calls=20)
            elif input_type == 5: # mfcc
                dataset = dataset.map(lambda x,y : (self._convert_to_mfcc(x), y), num_parallel_calls=20)               
                
                
            self._datasets.append(dataset)
        
        
    def _get_datasets(self):
    	return self._datasets
        
    def _get_dataset_placeholders(self):
        return self._placeholders
        
    def _get_data_lists(self):
        return self._data_lists, self._labels_lists


        
    '''
    def _num_total_batches(self):
        if self._data_size == 0:
            for list in self._data_lists:
                self._data_size += len(list)
    '''


    def _load_data_in_lists(self, dir_list):
        self._data_lists = []
        self._labels_lists = []
    
        for dir in dir_list:
            known_data, unknown_data = self._load_data_from_dir(self._train_data_dir)
            # shuffle unknown_data samples and take only a percentage of them
            np.random.shuffle(unknown_data)
            unknown_data = unknown_data[:int(len(known_data) * self._unknown_data_ratio)]
            
            # known_data is now the whole dataset
            known_data.extend(unknown_data)
            np.random.shuffle(known_data)
            
            wav_paths, label_list = zip(*known_data)
           
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                decoded_audios = list(executor.map(self._decode_wav_file, wav_paths))
            
            
            self._data_lists.append(np.array(decoded_audios))
            self._labels_lists.append(np.array(label_list))
        
        
        
    def _load_data_from_dir(self, data_dir):
        known_data = []
        unknown_data = []
        missing_words = set()
        
        # to find the most common known_data word
        known_data_word_occur = collections.Counter()
        
        # go over all WAV files in directory and sub-directories
        for wav_path in gfile.Glob(os.path.join(data_dir, '*', '*nohash*.wav')):
            # filter non-wav files
            if not wav_path.endswith(".wav") or "/".join(wav_path.split("/")[-2:]) in self._bad_paths:
                continue
            
            curr_word = wav_path.split("/")[-2].lower()
            
            # get encodings
            encoding = self._get_word_encoding(curr_word)
            if encoding:
                if curr_word in self._known_words:
                    known_data.append([wav_path, encoding])
                    known_data_word_occur[curr_word] += 1
                else:
                    unknown_data.append([wav_path, encoding])
            else:
                missing_words.add(curr_word)
        print("Words missing from the dataset:", missing_words)


        # make silence waves (all zeros) and add samples equal to number of occurances of the most common word
        silence_dir = os.path.join(data_dir, self._silence_alias)
        if not os.path.exists(silence_dir):
            os.makedirs(silence_dir)
        
        silence_file_path = os.path.join(silence_dir, "silence.wav")
        silence_encoded = np.zeros([self._silence_length], dtype=np.int16)
        scipy.io.wavfile.write(silence_file_path, 16000, silence_encoded)
        
        silence_encoding = self._get_word_encoding(self._silence_word)
        most_num_occurs = known_data_word_occur.most_common(1)[0][1]
        known_data.extend([[silence_file_path, silence_encoding]] * most_num_occurs)

        return known_data, unknown_data
        
     

    # if too slow, do it with tf map
    def _decode_wav_file(self, wav_path):
        sampling_rate, decoded_audio = scipy.io.wavfile.read(wav_path)
        decoded_audio = decoded_audio.astype(np.float32, copy=False)
        
        # bring into the 16-bit int range
        decoded_audio /= np.power(2, 15)
        curr_audio_length = len(decoded_audio)
        target_audio_length = int(self._audio_dur_in_ms * self._sampling_rate / 1000)

        # fix audio length by cutting or appending zeros equally from both ends
        start_index = abs(target_audio_length - curr_audio_length) // 2
        if curr_audio_length < target_audio_length:
            audio_reformatted = np.zeros(target_audio_length, dtype=np.float32)
            audio_reformatted[start_index : start_index + curr_audio_length] = decoded_audio
        elif curr_audio_length > target_audio_length:
            audio_reformatted = decoded_audio[start_index : start_index + target_audio_length]

        return audio_reformatted
        
    
    # word -> alias -> char encoding
    def _get_word_encoding(self, word):
        try:
            encoding = [self._char_to_id[char] for char in self._word_to_alias[word]]
            # add zeros to make all encoding equal length - allows to be packed in tensor
            encoding.extend([0 for i in range(self._label_encoding_length - len(encoding))])
            return encoding
        except KeyError:
            return None
    
    # char encoding -> alias -> label (specific known word / unknown)
    def _get_label_from_encoding(self, word_encoding):
        alias = ''.join([self._id_to_char[i] for i in word_encoding])
        try:
            label = self._alias_to_label[alias]
        except KeyError:
            return self._unknown_label
        return label
        
        
    def _convert_to_log_mel_specs(self, data_batch):
        # takes a batch of mono PCM samples
        # input data_batch (batch_size, audio_length)
        mel_spectrograms = self._convert_to_mel_specs(data_batch)     
        
        v_max = tf.reduce_max(mel_spectrograms, axis=[1, 2], keepdims=True)
        v_min = tf.reduce_min(mel_spectrograms, axis=[1, 2], keepdims=True)
        is_zero = tf.cast(tf.equal(v_max - v_min, 0), tf.float32)
        scaled_mel_specs = (mel_spectrograms - v_min) / (v_max - v_min + is_zero)

        epsilon = 0.001
        log_mel_specs = tf.log(scaled_mel_specs + epsilon)
        v_min = np.log(epsilon)
        v_max = np.log(epsilon + 1)
        
        scaled_log_mel_specs = (log_mel_specs - v_min) / (v_max - v_min)
            
        # (batch_size, num_frames=112, num_mel_spec_bins=46)
        return scaled_log_mel_specs
    
    
    def _convert_to_mel_specs(self, data_batch):
        # takes a batch of mono PCM samples
        # input data_batch (batch_size, audio_length)

        # get magnitude spectrograms via the short-term Fourier transform
        # (batch_size, num_frames,=112 num_mag_spec_bins)
        mag_spectrograms = self._convert_to_mag_specs(data_batch)
        num_mag_spec_bins = 1 + (self._frame_size // 2)

        # warp the linear scale to mel scale
        # [num_mag_spec_bins, num_mel_spec_bins]
        mel_weights = tf.contrib.signal.linear_to_mel_weight_matrix(self._num_spec_bins,
                                                                    num_mag_spec_bins, 
                                                                    self._sampling_rate,
                                                                    lower_edge_hertz=20.0, 
                                                                    upper_edge_hertz=4000.0)

        # convert the magnitude spectrograms to mel spectrograms 
        # (batch_size, num_frames, num_mel_spec_bins)
        mel_spectrograms = tf.tensordot(mag_spectrograms , mel_weights, 1)
        mel_spectrograms.set_shape([mag_spectrograms.shape[0], 
                                   mag_spectrograms.shape[1], 
                                   self._num_spec_bins])        
            
        # (batch_size, num_frames=112, num_mel_spec_bins=46)
        return mel_spectrograms
    
    def _convert_to_mag_specs(self, data_batch):
        # get magnitude spectrograms via the short-term Fourier transform
        # (batch_size, num_frames,=112 num_mag_spec_bins)
        mag_spectrograms = tf.abs(tf.contrib.signal.stft(data_batch,
                                                        frame_length=self._frame_size,
                                                        frame_step=self._frame_stride,
                                                        fft_length=self._frame_size))
        
        self._num_spec_bins = 1 + (self._frame_size // 2)
        return mag_spectrograms
        
    def _convert_to_mfcc(self, data_batch):
        return tf.contrib.signal.mfccs_from_log_mel_spectrograms(self._convert_to_log_mel_specs(data_batch))
    
    
    def _convert_to_mfcc(self, data_batch):
        data_batch = data_batch[:18144]
        data_batch.set_shape([self._num_frames, 162])
        return data_batch
    
    def _convert_to_log_mag_specs(self, data_batch):
        mag_specs = self._convert_to_mag_specs(data_batch)

        v_max = tf.reduce_max(mag_specs, axis=[1, 2], keepdims=True)
        v_min = tf.reduce_min(mag_specs, axis=[1, 2], keepdims=True)
        is_zero = tf.cast(tf.equal(v_max - v_min, 0), tf.float32)
        scaled_mel_specs = (mag_specs - v_min) / (v_max - v_min + is_zero)

        epsilon = 0.001
        log_mag_specs = tf.log(scaled_mel_specs + epsilon)
        v_min = np.log(epsilon)
        v_max = np.log(epsilon + 1)

        scaled_log_mag_specs = (log_mag_specs - v_min) / (v_max - v_min)
            
        # (batch_size, num_frames=112, num_mel_spec_bins=46)
        return scaled_log_mag_specs