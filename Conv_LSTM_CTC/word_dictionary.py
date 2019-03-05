import os
import glob
import tensorflow as tf
from tensorflow.python.platform import gfile
import csv
import numpy as np

class WordDict():
    '''
    WordDict class that provides word/character related methods, including
    converting original words to modified version (modi) (e.g. removing
    unpronounced characters), creating/keeping word and character dictionaries,
    calculating distribution of words in a dataset or a batch, converting
    words into character indices for training and the reverse for predictions.
    '''
    
    def __init__(self, file_words, file_chars, num_key_words, silence_class):
        self._word_to_modi = {}
        self._modi_to_word = {}
        self._word_to_modi[silence_class] = silence_class
        self._modi_to_word[silence_class] = "silence"
        self._all_words = []
        with gfile.GFile(file_words, 'r') as fp:
            reader = csv.reader(fp, delimiter=',')
            for row in reader:
                if not row[0].startswith('#'):
                    org = row[0]
                    train_w = row[1]
                    self._all_words.append(org)
                    self._word_to_modi[org] = train_w
                    self._modi_to_word[train_w] = org
        self._all_words += ["unknown", "silence"]

        train_modis = list(self._word_to_modi.values())
        self._num_word_classes = len(train_modis)
        self._max_label_length = max([len(w) for w in train_modis])

        chars = list(set(''.join(train_modis)))
        self._idx_to_char, self._char_to_idx = map_chars(file_chars, chars=chars)
        self._num_char_classes = len(self._idx_to_char)

        # modi_to_target dictionary
        self._key_words = self._all_words[:num_key_words]
        self._modi_to_target = {}
        for modi, word in self._modi_to_word.items():
            if word in self._key_words + ["silence"]:
                self._modi_to_target[modi] = word

        # word_to_target dictionary
        self._word_to_target = {}
        for word in self._all_words:
            if word in self._key_words + ["silence"]:
                self._word_to_target[word] = word
            else:
                self._word_to_target[word] = "unknown"

    def word_distro(self, words, msg='', verbose=False):
        """Calculates the distribution of words and characters.
        """
        w_cnt = collections.Counter()
        for w in words:
            w_cnt[self.indices_to_modi(w)] += 1

        tf.logging.debug('[%s] %d, %d, %s', msg, len(w_cnt),
                                         sum(w_cnt.values()), w_cnt)

        if verbose:
            c_cnt = collections.Counter()
            for w in list(w_cnt.keys()):
                for c in list(w):
                    c_cnt[c] += w_cnt[w]
            tf.logging.debug('[%s] %d, %s', msg, len(c_cnt), c_cnt)

            w_sum = sum(w_cnt.values())
            for key in w_cnt.keys():
                w_cnt[key] = round(128 * w_cnt[key] / w_sum, 1)
            tf.logging.debug('[%s] %d, %s', msg, len(w_cnt), w_cnt)

        return w_cnt

    @property
    def num_classes(self):
        return self._num_word_classes, self._num_char_classes

    @property
    def max_label_length(self):
        return self._max_label_length

    @property
    def key_words(self):
        return self._key_words

    @property
    def all_words(self):
        return self._all_words

    def word_to_indices(self, word):
        """Converts original words to modified version(modi), then to character
            indices for training.
        """
        try:
            modi = self._word_to_modi[word]
            indices = [self._char_to_idx[c] for c in modi]
            # pad the rest of the list
            indices.extend([0 for i in range(self._max_label_length - len(indices))])
            return indices
        except KeyError:
            return None

    def indices_to_modi(self, indices):
        """Converts character indices to modified version of words, used for
            training/ validation debugging.
        """
        modi = ''.join([self._idx_to_char[i] for i in indices])
        return modi

    def modi_to_word(self, modi):
        """Converts modified version of words to the original words."""
        try:
            word = self._modi_to_word[modi]
        except KeyError:
            word = "unknown"
        return word

    def indices_to_submit(self, indices):
        """Converts character indices to key words (including silence) and unknown
            words for submission.
        """
        modi = ''.join([self._idx_to_char[i] for i in indices])
        try:
            word = self._modi_to_target[modi]
        except KeyError:
            word = "unknown"
        return word

    def word_to_submit(self, word):
        """Converts non-key works to 'unknown' for submission."""
        return self._word_to_target[word]
        
        
def map_chars(file_chars, chars=None):
    '''
    Creates character-index mapping. The mapping needs to be constant for
    training and inference.
    '''
    if not os.path.exists(file_chars):
        tf.logging.info('WARNING!!!! regenerating %s', file_chars)
        idx_to_char = {i + 1: c for i, c in enumerate(chars)}
        # 0 is not used, dense to sparse array
        idx_to_char[0] = ''
        # null label
        idx_to_char[len(idx_to_char)] = '_'

        with gfile.GFile(file_chars, 'w') as fp:
            for i, c in idx_to_char.items():
                fp.write('%d,%s\n' % (i, c))
    else:
        with gfile.GFile(file_chars, 'r') as fp:
            reader = csv.reader(fp, delimiter=',')
            idx_to_char = {int(i): c for i, c in reader}

    char_to_idx = {c: i for i, c in idx_to_char.items()}
    return idx_to_char, char_to_idx