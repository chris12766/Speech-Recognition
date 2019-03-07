
silence_word = "silence"
silence_alias = "9"
unknown_word = "unknown"

max_label_length = 4    
# the top num_known_words in map_words.txt are key words
num_known_words = 10
word_to_alias = {"yes" : "yes",
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
                silence_word : silence_alias}
alias_to_word = {v: k for k, v in word_to_alias.items()}

#aliases = [*alias_to_word]
all_words = [*word_to_alias].extend([unknown_word])
known_words = all_words[:num_known_words]

# used to classify
#word_to_label = {}
alias_to_label = {}
for alias, word in alias_to_word.items():
    if word in known_words or word == silence_word:
        #word_to_label[word] = word
        alias_to_label[alias] = word
    #else:
    #    word_to_label[word] = unknown_word


id_to_char = {0 , "",
            1 , "U",
            2 , "9",
            3 , "m",
            4 , "r",
            5 , "o",
            6 , "b",
            7 , "I",
            8 , "y",
            9 , "d",
            10 , "l",
            11 , "h",
            12 , "c",
            13 , "e",
            14 , "g",
            15 , "n",
            16 , "O",
            17 , "f",
            18 , "u",
            19 , "p",
            20 , "t",
            21 , "w",
            22 , "v",
            23 , "E",
            24 , "s",
            25 , "i",
            26 , "a",
            27 , "_"}


# char encoding -> alias -> label (specific known word / unknown)
def get_label_from_encoding(word_encoding):
    alias = ''.join([id_to_char[i] for i in encoding])
    try:
        label = alias_to_label[alias]
    except KeyError:
        return unknown_word
    return label


