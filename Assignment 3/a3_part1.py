import os, string, re
import codecs
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

ds_path = './hwk3_datasets/'
ds = os.listdir(ds_path)

sets = ['IMDB-', 'yelp-']
types = ['train.txt', 'valid.txt', 'test.txt', ]
vocab_lists = {}

datasets = []

def preprocess(text):
    text = text.lower().replace('<br /><br />', ' ').replace('\t', ' ')
    return re.sub('[' + string.punctuation + ']', ' ', text)

''' training sets '''
for set in sets[1:]:
    ds = []
    with open(ds_path + set + types[0], 'r', encoding="utf-8") as f:
        file = f.read()
    file = preprocess(file)
    file.replace('0\n', ' ').replace('1\n', ' ')
    counter = Counter(file.split(" "))

    vectorizer = CountVectorizer(max_features=10000)
    vectorizer.fit_transform([file])
    vocab = vectorizer.vocabulary_
    vocab_lists[set] = vocab.keys()

    writer = open(set.split('-')[0] + '-vocab.txt', 'w')

    for key, value in vocab.items():
        text = "{} {} {}\n".format(key, value, counter[key])
        writer.write(text)


    ''' all the files '''
    for type in types:
        print(ds_path + set + types[0])
        with open(ds_path + set + types[0], 'r', encoding="utf-8") as f:
            file = f.read()
        
        file = preprocess(file)

        examples = file.split('\n')[:-1]
        ds_output = [i[-1] for i in examples]

        writer = open(set.split('-')[0] + '-' + type.split('.')[0] + '-vocab.txt', 'w')

        for i in range(len(examples)):
            text = ""
            for word in examples[i].split(' ')[:-1]:
                if word in vocab_lists[set]: 
                    text = "{} {}".format(text, vocab[word])

            text = "{}\t{}\n".format(text, ds_output[i])
            writer.write(text)


        
        

