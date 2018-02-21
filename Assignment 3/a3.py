import os, string, re, codecs, random
import scipy
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

ds_path = './hwk3_datasets/'
types = ['train.txt', 'valid.txt', 'test.txt', ]

n = 10000

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

def train_models(name, set):
    train = set['train']
    valid = set['valid']
    test = set['test']

    train_input = np.array(train[0])
    valid_input = np.array(valid[0])
    test_input = np.array(test[0])

    train_truth = np.array(train[1])
    valid_truth = np.array(valid[1])
    test_truth = np.array(test[1])

    classes = len(np.unique(train_truth))
    average = None if (classes > 2) else 'binary'
    # Random Uniform Classifier
    pred = np.rint(np.random.random(len(train_truth)) * (classes - 1))
    print("{} Random Uniform Classifier train f1_score {}".format(name, f1_score(train_truth, pred, average = average)))

    pred = np.rint(np.random.random(len(valid_truth)) * (classes - 1))
    print("{} Random Uniform Classifier valid f1_score {}".format(name, f1_score(valid_truth, pred, average = average)))

    pred = np.rint(np.random.random(len(test_truth)) * (classes - 1))
    print("{} Random Uniform Classifier test f1_score {}\n".format(name, f1_score(test_truth, pred, average = average)))


    # Majority Class Classifier
    maj = np.argmax(np.bincount(train_truth))

    pred = np.array([maj for i in range(len(train_truth))])
    print("{} Majority Class Classifier trian f1_score {}".format(name, f1_score(train_truth, pred, average = average)))

    pred = np.array([maj for i in range(len(valid_truth))])
    print("{} Majority Class Classifier valid f1_score {}".format(name, f1_score(valid_truth, pred, average = average)))

    pred = np.array([maj for i in range(len(test_truth))])
    print("{} Majority Class Classifier test f1_score {}\n".format(name, f1_score(test_truth, pred, average = average)))


    # Naive Bayes
    alpha = [i * 0.01 for i in range(10)]
    tuned_parameters = [{'alpha': alpha}]

    n_folds = 5

    clf = MultinomialNB()
    clf = GridSearchCV(clf, tuned_parameters, cv=n_folds, refit=False)
    clf.fit(train_input, train_truth)
    scores = np.array(clf.cv_results_['mean_test_score'])
    best_alpha = alpha[np.argmax(scores)]

    clf = MultinomialNB(alpha=best_alpha) if classes > 2 else GaussianNB()
    clf.fit(train_input, train_truth)

    pred = clf.predict(train_input)
    print("{} Naive Bayes Classifier train f1_score {}, alpha used = {}".format(name, f1_score(train_truth, pred, average = average), best_alpha))

    pred = clf.predict(valid_input)
    print("{} Naive Bayes Classifier valid f1_score {}, alpha used = {}".format(name, f1_score(valid_truth, pred, average = average), best_alpha))

    pred = clf.predict(test_input)
    print("{} Naive Bayes Classifier test f1_score {}, alpha used = {}\n".format(name, f1_score(test_truth, pred, average = average), best_alpha))
    print(clf.get_params(), "\n")




    # # Decision Tree
    # best_depth = 1
    # best_f1 = 0
    # for max_depth in range(3, 20):
    #     clf = DecisionTreeClassifier(max_depth=max_depth)
    #     clf.fit(train_input, train_truth)
    #     pred = clf.predict(valid_input)
    #     f1 = np.mean(f1_score(valid_truth, pred, average = average))
    #     if f1 > best_f1 : 
    #         # print("update best_depth to ", max_depth)
    #         best_depth = max_depth
    #         best_f1 = f1

    # clf = DecisionTreeClassifier(max_depth=best_depth)
    # clf.fit(train_input, train_truth)

    # pred = clf.predict(train_input)
    # print("{} Decision Tree Classifier train f1_score {}, max_depth used = {}".format(name, f1_score(train_truth, pred, average = average), best_depth))

    # pred = clf.predict(valid_input)
    # print("{} Decision Tree Classifier valid f1_score {}, max_depth used = {}".format(name, f1_score(valid_truth, pred, average = average), best_depth))

    # pred = clf.predict(test_input)
    # print("{} Decision Tree Classifier test f1_score {}, max_depth used = {}\n".format(name, f1_score(test_truth, pred, average = average), best_depth))


    # #Linear SVM 
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

    # clf = LinearSVC()
    # clf.fit(train_input, train_truth)

    # pred = clf.predict(train_input)
    # print("{} Linear SVM Classifier train f1_score {}".format(name, f1_score(train_truth, pred, average = average)))

    # pred = clf.predict(valid_input)
    # print("{} Linear SVM Classifier valid f1_score {}".format(name, f1_score(valid_truth, pred, average = average)))

    # pred = clf.predict(test_input)
    # print("{} Linear SVM Classifier test f1_score {}\n ".format(name, f1_score(test_truth, pred, average = average)))


def preprocess(file):
    with open(file, 'r', encoding="utf-8") as f:
        text = f.read()
    text = text.lower().replace('\t', ' ').replace('<br /><br />', ' ').strip()
    return re.compile('[^\w\s]').sub('', text)


''' DATA PARSING '''
def feature_extraction(set):
    file = preprocess(ds_path + set + types[0])
    word_list = list(filter(None, file.split(" "))) 
    counter = Counter(word_list).most_common(n)

    dict = {}

    writer = open(set.split('-')[0] + '-vocab.txt', 'w')

    # save top words
    for i in range(n):
        word = counter[i][0]
        dict[word] = i + 1
        
        # text = ("{}\t{}\t{}\n".format(word, i + 1, counter[i][1]))
        # writer.write(text)

    for type in types:
        print(ds_path + set + type)
        file = preprocess(ds_path + set + type)

        examples = list(filter(None, file.split("\n"))) 
        ds_output = [i[-1] for i in examples]

        # writer = open(set.split('-')[0] + '-' + type.split('.')[0] + '-vocab.txt', 'w')
        # for i in range(len(examples)):
        #     text = ""
        #     for word in examples[i].split(' ')[:-1]:
        #         if word in dict.keys(): 
        #             text = "{} {}".format(text, dict[word])
        #     if len(text) == 0: text = ' '
        #     text = "{}\t{}\n".format(text, ds_output[i])
        #     writer.write(text[1:])

    return dict

def get_bow(dict, set):
    bow = {}
    bow_f = {}
    for type in types: 
        name = type.split('.')[0]
        text  = preprocess(ds_path + set + type).split('\n')

        text = list(filter(None, text))

        output = [int(line[-1]) for line in text]
        examples = [line[:-1] for line in text]

        vectorizer = CountVectorizer(vocabulary = dict.keys())

        vectors = np.asarray(vectorizer.fit_transform(examples).todense())

        freq = normalize(vectors)
        vectors[vectors > 1] = 1
        binary = vectors

        bow[name] = [binary, output]
        bow_f[name] = [freq, output]

    return bow, bow_f

if __name__:
    sets = ['yelp-', 'IMDB-']

    #============== yelp ================
    set = sets[0]
    vocab_list = feature_extraction(set)
    yelp_bow, yelp_bowf = get_bow(vocab_list, set)

    print("Using the BINARY Bag of Words")
    train_models(set, yelp_bow)

    print("Using the Frequency Bag of Words")
    train_models(set, yelp_bowf)



    #============== IMDB ================
    set = sets[1]
    vocab_list = feature_extraction(set)
    IMDB_bow, IMDB_bowf = get_bow(vocab_list, set)

    print("Using the BINARY Bag of Words")
    train_models(set, IMDB_bow)

    print("Using the Frequency Bag of Words")
    train_models(set, IMDB_bowf)


