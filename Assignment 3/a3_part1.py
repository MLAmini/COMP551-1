import os, string, re, codecs, random
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

ds_path = './hwk3_datasets/'
ds = os.listdir(ds_path)

sets = ['yelp-', 'IMDB-']
types = ['train.txt', 'valid.txt', 'test.txt', ]
vocab_lists = {}

bow = {} 
bow_f = {}

a = np.array([1, 0, 1, 1, 1, 1, 0, 0, 0])
b = np.array([0, 0, 1, 0, 1, 1, 1, 0, 0])

'''
bow = {
    'IMDB': {
        'train': [[bag of word], [output]]
        'valid': [[bag of word], [output]]
        'test': [[bag of word], [output]]
    }
    'yelp':, 
        'train': [[bag of word], [output]]
        'valid': [[bag of word], [output]]
        'test': [[bag of word], [output]]
}
'''


def confusion_matrix(pred, truth):
    # m = [[tn, fp], [fn, tp]]
    m = [[0, 0], [0, 0]]
    tp = np.dot(pred, truth)
    diff = pred - truth
    m[0][0] = np.count_nonzero(diff == 0) - tp
    m[0][1] = np.count_nonzero(diff == 1)
    m[1][0] = np.count_nonzero(diff == -1)
    m[1][1] = tp

    return m

def preprocess(text):
    text = text.lower().replace('<br /><br />', ' ').replace('\t', ' ')
    return re.sub('[' + string.punctuation + ']', ' ', text)

''' training sets '''
for set in sets[:1]:
    with open(ds_path + set + types[0], 'r', encoding="utf-8") as f:
        file = f.read()
    file = preprocess(file)
    file.replace('0\n', ' ').replace('1\n', ' ')
    counter = Counter(file.split(" "))

    vectorizer = CountVectorizer(max_features=10000)
    vectorizer.fit_transform([file])
    vocab = vectorizer.vocabulary_
    vocab_lists[set] = vocab.keys()

    # writer = open(set.split('-')[0] + '-vocab.txt', 'w')

    # for key, value in vocab.items():
    #     text = "{}\t{}\t{}\n".format(key, value, counter[key])
    #     writer.write(text)


    ''' all the files '''
    bow_ds = {}
    bowf_ds = {}
    for type in types:
        print(ds_path + set + type)
        with open(ds_path + set + type, 'r', encoding="utf-8") as f:
            file = f.read()
        
        file = preprocess(file)


        examples = file.split('\n')[:-1]
        ds_output = [i[-1] for i in examples]
        examples = [i[:-1] for i in examples]

        vectorizer = CountVectorizer(max_features=10000)
        bow_ds[type.split('.')[0]] = [vectorizer.fit_transform(examples).todense(), ds_output]

        vectorizer = TfidfVectorizer(max_features=10000, use_idf=False, norm='l1')
        bowf_ds[type.split('.')[0]] = [vectorizer.fit_transform(examples).todense(), ds_output]


        # writer = open(set.split('-')[0] + '-' + type.split('.')[0] + '-vocab.txt', 'w')

        # for i in range(len(examples)):
        #     text = ""
        #     for word in examples[i].split(' ')[:-1]:
        #         if word in vocab_lists[set]: 
        #             text = "{} {}".format(text, vocab[word])

        #     text = "{}\t{}\n".format(text, ds_output[i])
        #     writer.write(text[1:])

    bow[set] = bow_ds
    bow_f[set] = bowf_ds

for bag_of_words in [bow, bow_f]:
    for set in bag_of_words.items():
        print('\nData set ', set[0].split('-')[0], "\n")
        train = set[1]['train']
        valid = set[1]['valid']
        test = set[1]['test']

        train_input = np.array(train[0]).astype(int)
        valid_input = np.array(valid[0]).astype(int)
        test_input = np.array(test[0]).astype(int)

        train_truth = np.array(train[1]).astype(int) 
        valid_truth = np.array(valid[1]).astype(int) 
        test_truth = np.array(test[1]).astype(int) 

        classes = len(np.unique(test_truth))
        average = None if (classes > 2) else 'binary'

        # Random Uniform Classifier
        pred = np.rint(np.random.random(len(train_truth)) * (classes - 1))
        print("{} Random Uniform Classifier train f1_score {}".format(set[0], f1_score(train_truth, pred, average = average)))

        pred = np.rint(np.random.random(len(valid_truth)) * (classes - 1))
        print("{} Random Uniform Classifier valid f1_score {}".format(set[0], f1_score(valid_truth, pred, average = average)))

        pred = np.rint(np.random.random(len(test_truth)) * (classes - 1))
        print("{} Random Uniform Classifier test f1_score {}\n".format(set[0], f1_score(test_truth, pred, average = average)))


        # Majority Class Classifier
        maj = np.argmax(np.bincount(train_truth))
        
        pred = np.array([maj for i in range(len(train_truth))])
        print("{} Majority Class Classifier trian f1_score {}".format(set[0], f1_score(train_truth, pred, average = average)))

        pred = np.array([maj for i in range(len(valid_truth))])
        print("{} Majority Class Classifier valid f1_score {}".format(set[0], f1_score(valid_truth, pred, average = average)))

        pred = np.array([maj for i in range(len(test_truth))])
        print("{} Majority Class Classifier test f1_score {}\n".format(set[0], f1_score(test_truth, pred, average = average)))


        # Naive Bayes
        best_alpha = 0.5
        best_f1 = 0
        for alpha in range(5):
            clf = MultinomialNB(alpha=alpha) if classes > 2 else BernoulliNB(alpha=alpha)
            clf.fit(train_input, train_truth)
            pred = clf.predict(valid_input)
            f1 = np.mean(f1_score(valid_truth, pred, average = average))
            if f1 > best_f1 : 
                # print("update best_alpha to ", alpha)
                best_alpha = alpha
                best_f1 = f1

        clf = MultinomialNB(alpha=best_alpha) if classes > 2 else BernoulliNB(alpha=best_alpha)
        clf.fit(train_input, train_truth)

        pred = clf.predict(train_input)
        print("{} Naive Bayes Classifier train f1_score {}, alpha used = {}".format(set[0], f1_score(train_truth, pred, average = average), best_alpha))

        pred = clf.predict(valid_input)
        print("{} Naive Bayes Classifier valid f1_score {}, alpha used = {}".format(set[0], f1_score(valid_truth, pred, average = average), best_alpha))

        pred = clf.predict(test_input)
        print("{} Naive Bayes Classifier test f1_score {}, alpha used = {}\n".format(set[0], f1_score(test_truth, pred, average = average), best_alpha))

        # Decision Tree
        best_depth = 0
        best_f1 = 0
        for max_depth in range(3, 20):
            clf = DecisionTreeClassifier(max_depth=max_depth)
            clf.fit(train_input, train_truth)
            pred = clf.predict(valid_input)
            f1 = np.mean(f1_score(valid_truth, pred, average = average))
            if f1 > best_f1 : 
                # print("update best_depth to ", max_depth)
                best_depth = max_depth
                best_f1 = f1

        clf = DecisionTreeClassifier(max_depth=best_depth)
        clf.fit(train_input, train_truth)

        pred = clf.predict(train_input)
        print("{} Decision Tree Classifier train f1_score {}, max_depth used = {}".format(set[0], f1_score(train_truth, pred, average = average), best_depth))

        pred = clf.predict(valid_input)
        print("{} Decision Tree Classifier valid f1_score {}, max_depth used = {}".format(set[0], f1_score(valid_truth, pred, average = average), best_depth))

        pred = clf.predict(test_input)
        print("{} Decision Tree Classifier test f1_score {}, max_depth used = {}\n".format(set[0], f1_score(test_truth, pred, average = average), best_depth))


        #Linear SVM 
        # best_depth = 0
        # best_f1 = 0
        # for max_depth in range(3, 20):
        #     clf = LinearSVC()
        #     clf.fit(train_input, train_truth)
        #     pred = clf.predict(valid_input)
        #     f1 = np.mean(f1_score(valid_truth, pred, average = average))
        #     if f1 > best_f1 : 
        #         # print("update best_depth to ", max_depth)
        #         best_depth = max_depth
        #         best_f1 = f1

        clf = LinearSVC()
        clf.fit(train_input, train_truth)

        pred = clf.predict(train_input)
        print("{} Linear SVM Classifier train f1_score {}, max_depth used = {}".format(set[0], f1_score(train_truth, pred, average = average), best_depth))

        pred = clf.predict(valid_input)
        print("{} Linear SVM Classifier valid f1_score {}, max_depth used = {}".format(set[0], f1_score(valid_truth, pred, average = average), best_depth))

        pred = clf.predict(test_input)
        print("{} Linear SVM Classifier test f1_score {}, max_depth used = {}\n ".format(set[0], f1_score(test_truth, pred, average = average), best_depth))




