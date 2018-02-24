import os, string, re, codecs, random
from collections import Counter
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from sklearn.model_selection import GridSearchCV
from sklearn.cross_validation import PredefinedSplit


def warn(*args, **kwargs):
	pass
import warnings
warnings.warn = warn

types = ['train.txt', 'valid.txt', 'test.txt', ]
average = 'micro'


def preprocess(file):
	"""
	Keyword arguments: 
	file -- string file path (string)

	Returns:
	processed file (string)

	the function reads the file, puts everything in lower case and removes punctuation marks 
	"""
	translator = str.maketrans(" ", " ", string.punctuation)
	with open(file, 'r', encoding="utf-8") as f:
		text = f.read()
	text = text.lower().replace('\t', ' ').replace('<br /><br />', ' ').translate(translator)
	return text

def feature_extraction(name, n):
	"""
	Keyword arguments: 
	name -- set name (IMBD or yelp) - string
	n -- number of features - int

	Returns: 
	dictionary {"words": ID}

	the function
	- extracts the top n frequent features with their respective ID. 
	- writes output file for feature ID and count 
	- writes output files for feature vectors for train, valid, test set 
	"""
	file = preprocess(ds_path + name + types[0])
	word_list = file.split(" ")
	counter = Counter(word_list).most_common(n) #count the occurence of each word in the text
	dict = {}


	# save the top features in a output file "name-vocab.txt" 
	# "word" id  count
	writer = open(name.split('-')[0] + '-vocab.txt', 'w')

	for i in range(n):
		word = counter[i][0]
		dict[word] = i + 1
		
		text = ("{}\t{}\t{}\n".format(word, i + 1, counter[i][1]))
		writer.write(text)

	# write feature vectors for every sample in train, valid and test sets 
	for type in types:
		print(ds_path + name + type)
		file = preprocess(ds_path + name + type)

		examples = file.split("\n")[:-1]
		ds_output = [i[-1] for i in examples]

		writer = open(name.split('-')[0] + '-' + type.split('.')[0] + '.txt', 'w')
		for i in range(len(examples)):
		    text = ""
		    for word in examples[i].split(' ')[:-1]:
		        if word in dict.keys(): 
		            text = "{} {}".format(text, dict[word])
		    if len(text) == 0: text = ' '
		    text = "{}\t{}\n".format(text, ds_output[i])
		    writer.write(text[1:])

	return dict

def get_bow(dict, set):
	""""
	Keyword arguments: 
	dict -- top features vocabulary (dict)
	set -- name of the set (string) -- IMDB or yelp

	Returns: 
	[binary bow vectors, truth], [frequency bow vectors, truth]

	the function: 
	- uses CountVectorizer and the feature vocabulary to construct bag of word vectors
	"""
	bow = {}
	bow_f = {}
	for type in types: 
		name = type.split('.')[0]
		text  = preprocess(ds_path + set + type).split('\n')

		text = list(filter(None, text))

		output = np.array([int(line[-1]) for line in text])
		examples = [line[:-1] for line in text]

		vectorizer = CountVectorizer(vocabulary = dict.keys())

		vectors = np.asarray(vectorizer.fit_transform(examples).todense())

		#save freq and binary as sparse vectors for faster training
		freq = sparse.csr_matrix(normalize(vectors))
		vectors[vectors > 1] = 1 #set all count > 1 to 1, to binarize the vector
		binary = sparse.csr_matrix(vectors)

		bow[name] = [binary, output]
		bow_f[name] = [freq, output]

	return bow, bow_f

def train_model(set, clf, params):
	""""
	Keyword arguments: 
	set -- dataset (dictionary)
	clf -- sklearn model 
	params -- fine-tuning parameter 

	Returns: 
	f1_score train, f1_score valid, f1_score test, best parameter 

	the function: 
	- uses GridSearchCV to find the best hyperparameter for the model
	- refits the model with the parameters 
	- predicts train, valid and test sets 
	- find respective f1_scores
	"""
	train = set['train']
	valid = set['valid']
	test = set['test']

	train_input = train[0]
	valid_input = valid[0]
	test_input = test[0]

	train_truth = train[1]
	valid_truth = valid[1]
	test_truth = test[1]

	if params != None:
		combine_input = sparse.vstack([train_input, valid_input])
		combine_truth = np.concatenate((train_truth, valid_truth))
		fold = [-1 for i in range(train_input.shape[0])] + [0 for i in range(valid_input.shape[0])]
		ps = PredefinedSplit(test_fold = fold)
		clf = GridSearchCV(clf, params, cv=ps, refit=True)
		clf.fit(combine_input, combine_truth)
	else:
		clf.fit(train_input, train_truth)

	best_param = None if params==None else clf.best_params_
	
	f1_train = f1_score(train_truth, clf.predict(train_input), average = average)
	f1_valid = f1_score(valid_truth, clf.predict(valid_input), average = average)
	f1_test = f1_score(test_truth, clf.predict(test_input), average = average)

	return f1_train, f1_valid, f1_test, best_param

def random_class(set):
	""""
	Keyword arguments: 
	set -- dataset (dictionary)

	Returns: 
	f1_score train, f1_score valid, f1_score test
	"""
	train_truth = set['train'][1]
	valid_truth = set['valid'][1]
	test_truth = set['test'][1]

	classes = len(np.unique(train_truth))

	# predict the output of each set randomly through all classes
	f1_train = f1_score(train_truth, np.rint(np.random.random(len(train_truth)) * (classes - 1)), average = average)
	f1_valid = f1_score(valid_truth, np.rint(np.random.random(len(valid_truth)) * (classes - 1)), average = average)
	f1_test = f1_score(test_truth, np.rint(np.random.random(len(test_truth)) * (classes - 1)), average = average)
	return f1_train, f1_valid, f1_test

def majority_class(set):
	""""
	Keyword arguments: 
	set -- dataset (dictionary)

	Returns: 
	f1_score train, f1_score valid, f1_score test
	"""

	train_truth = set['train'][1]
	valid_truth = set['valid'][1]
	test_truth = set['test'][1]

	#find the class the majority class
	maj = np.argmax(np.bincount(train_truth))

	#predict the output of every sample to be = majority class
	f1_train = f1_score(train_truth, np.array([maj for i in range(len(train_truth))]), average = average)
	f1_valid = f1_score(valid_truth, np.array([maj for i in range(len(valid_truth))]), average = average)
	f1_test = f1_score(test_truth, np.array([maj for i in range(len(test_truth))]), average = average)

	return f1_train, f1_valid, f1_test


if __name__:
	n = 10000
	ds_path = './hwk3_datasets/'
	sets = ['yelp-', 'IMDB-']
	average = 'micro'


	#============== yelp ================
	set = sets[0]
	vocab_list = feature_extraction(set, n)

	yelp_bow, yelp_bowf = get_bow(vocab_list, set)

	# print("\nBINARY YELP\n")


	# pred = random_class(yelp_bow)
	# print(set, "Random Classifier \n(train, valid, test) = ", pred)

	# pred = majority_class(yelp_bow)
	# print(set, "Majority Classifier \n(train, valid, test) = ", pred)

	# param = [{'alpha': np.arange(0.6, 0.8, 0.01)}]
	# pred = train_model(yelp_bow, BernoulliNB(), param)
	# print(set, "Naive Bayes Classifier \n(train, valid, test) = ", pred[:3])
	# print("best params = {}\n".format(pred[3]))

	# param = [{'max_depth': [i for i in range(10, 20)], 'max_features': [1000 * i for i in range(2, 7)], 'max_leaf_nodes': [1000 * i for i in range(3, 6)]}]
	# pred = train_model(yelp_bow, DecisionTreeClassifier(), param)
	# print(set, "Decision Tree \n(train, valid, test) = ", pred[:3])
	# print("best params = {}\n".format(pred[3]))

	# param = [{'max_iter': [500 * i for i in range(5)]}]
	# pred = train_model(yelp_bow, LinearSVC(), param)
	# print(set, "Linear SVM Classifier \n(train, valid, test) = ", pred[:3])
	# print("best params = {}".format(pred[3]))


	# print("\nFREQUENCY YELP\n")

	# pred = random_class(yelp_bowf)
	# print(set, "Random Classifier \n(train, valid, test) = ", pred)

	# pred = majority_class(yelp_bowf)
	# print(set, "Majority Classifier \n(train, valid, test) = ", pred)

	# param = [{'max_depth': [i for i in range(10, 20)], 'max_features': [1000 * i for i in range(2, 7)], 'max_leaf_nodes': [1000 * i for i in range(3, 6)]}]
	# pred = train_model(yelp_bowf, DecisionTreeClassifier(), param)
	# print(set, "Naive Bayes Classifier \n(train, valid, test) = ", pred[:3])
	# print("best params = {}\n".format(pred[3]))

	# param = [{'max_iter': [500 * i for i in range(5)]}]
	# pred = train_model(yelp_bowf, LinearSVC(), param)
	# print(set, "Linear SVM Classifier \n(train, valid, test) = ", pred[:3])
	# print("best params = {}".format(pred[3]))

	yelp_bowf['train'][0] = yelp_bowf['train'][0].todense()
	yelp_bowf['valid'][0] = yelp_bowf['valid'][0].todense()
	yelp_bowf['test'][0] = yelp_bowf['test'][0].todense()

	pred = train_model(yelp_bowf, GaussianNB(), None)
	print(set, "Naive Bayes\n(train, valid, test) = ", pred[:3])





	# ============== IMDB ================
	set = sets[1]
	vocab_list = feature_extraction(set, n)
	IMDB_bow, IMDB_bowf = get_bow(vocab_list, set)

	print("\nBINARY IMBDF\n")

	pred = random_class(IMDB_bow)
	print(set, "Random Classifier \n(train, valid, test) = ", pred)

	pred = majority_class(IMDB_bow)
	print(set, "Majority Classifier \n(train, valid, test) = ", pred)


	param = [{'alpha': np.arange(0.6, 0.8, 0.01)}]
	pred = train_model(IMDB_bow, BernoulliNB(), param)
	print(set, "Naive Bayes Classifier \n(train, valid, test) = ", pred[:3])
	print("best params = {}\n".format(pred[3]))

	param = [{'max_depth': [i for i in range(10, 20)], 'max_features': [1000 * i for i in range(2, 7)], 'max_leaf_nodes': [1000 * i for i in range(3, 6)]}]
	pred = train_model(IMDB_bow, DecisionTreeClassifier(), param)
	print(set, "Decision Tree \n(train, valid, test) = ", pred[:3])
	print("best params = {}\n".format(pred[3]))

	param = [{'max_iter': [500 * i for i in range(5)]}]
	pred = train_model(IMDB_bow, LinearSVC(), param)
	print(set, "Linear SVM Classifier \n(train, valid, test) = ", pred[:3])
	print("best params = {}".format(pred[3]))

	print("\nFREQUENCY IMBDF\n")

	pred = random_class(IMDB_bowf)
	print(set, "Random Classifier \n(train, valid, test) = ", pred)

	pred = majority_class(IMDB_bowf)
	print(set, "Majority Classifier \n(train, valid, test) = ", pred)

	param = [{'max_depth': [i for i in range(10, 20)], 'max_features': [1000 * i for i in range(2, 7)], 'max_leaf_nodes': [1000 * i for i in range(3, 6)]}]
	pred = train_model(IMDB_bowf, DecisionTreeClassifier(), param)
	print(set, "Decision Tree \n(train, valid, test) = ", pred[:3])
	print("best params = {}\n".format(pred[3]))

	param = [{'max_iter': [500 * i for i in range(5)]}]
	pred = train_model(IMDB_bowf, LinearSVC(), param)
	print(set, "Linear SVM Classifier \n(train, valid, test) = ", pred[:3])
	print("best params = {}".format(pred[3]))

	IMDB_bowf['train'][0] = IMDB_bowf['train'][0].todense()
	IMDB_bowf['valid'][0] = IMDB_bowf['valid'][0].todense()
	IMDB_bowf['test'][0] = IMDB_bowf['test'][0].todense()

	pred = train_model(IMDB_bowf, GaussianNB(), None)
	print(set, "Naive Bayes Classifier \n(train, valid, test) = ", pred[:3])



