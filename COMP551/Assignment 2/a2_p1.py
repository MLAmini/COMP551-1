import pandas
import numpy
import matplotlib.pyplot as plt
import random
import scipy
import math


#Parse covariance matrix data
cov = pandas.read_csv('./dataset/DS1_Cov.txt', header=None)
cov.drop([20], axis=1, inplace=True)
cov = cov.values

#Parse the mean vectors
m0 = numpy.genfromtxt('./dataset/DS1_m_0.txt', delimiter=',')[:-1]
m1 = numpy.genfromtxt('./dataset/DS1_m_1.txt', delimiter=',')[:-1]

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                                    Part 1
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# 1) generate 2000 examples and split data

ds1 = []
ds0 = []
for i in range(2000):
    ds1.append(numpy.random.multivariate_normal(m1, cov))
    ds0.append(numpy.random.multivariate_normal(m0, cov))
ds1 = pandas.DataFrame(ds1)
ds0 = pandas.DataFrame(ds0)
ds0[20] = 0
ds1[20] = 1

msk = numpy.random.rand(len(ds1)) <= 0.7
train1 = ds1[msk]
test1 = ds1[~msk]

msk = numpy.random.rand(len(ds0)) <= 0.7
train0 = ds0[msk]
test0 = ds0[~msk]

train = pandas.concat([train0, train1], ignore_index=True)
test = pandas.concat([test0, test1], ignore_index=True)

train1.to_csv('DS1_train1.csv')
test1.to_csv('DS1_test1.csv')
train0.to_csv('DS1_train0.csv')
test0.to_csv('DS1_test0.csv')

test.to_csv('DS1_test.csv')
train.to_csv('DS1_train.csv')


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                2) LDA model using maximum likelihood approach
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

len0 = len(train0)
len1 = len(train1)


# estimated max probability
prob_0 = float(len0) / float(len0 + len1)
prob_1 = 1.0 - prob_0


# drop the ouput column
train0 = train[train[20] == 1]
train0 = train0[train0.columns.difference([20])]

train1 = train[train[20] == 1]
train1 = train1[train1.columns.difference([20])]

train01 = train0[1]

mean_0 = numpy.array((train0.mean())/len0)
mean_1 = numpy.array((train1.mean())/len1)

diff_0 = numpy.array(train0 - mean_0)
diff_1 = numpy.array(train1 - mean_1)

coeff_0 = numpy.sum(row.dot(row.T) for row in diff_0)
coeff_1 = numpy.sum(row.dot(row.T) for row in diff_1)

coeff = (coeff_0 + coeff_1) / float(len0 + len1)

w0 = math.log(prob_0) - math.log(prob_1) - 1/2 / coeff * (mean_0.dot(mean_0) - mean_1.dot(mean_1))
w1 = coeff * (mean_0 - mean_1)


print("w0: " + str(w0))
print("w1: " + str([i for i in w1]) + "\n")

test_output = numpy.array(test[20])
test.drop([20], inplace=True, axis=1)

dec_bound = numpy.matmul(test, w1) + w0

# replace dec_bound to 0 and 1
dec_bound[dec_bound < 0] = 0.0
dec_bound[dec_bound > 0] = 1.0

error_matrix = dec_bound - test_output

elem, cnt = numpy.unique(error_matrix, return_counts=True)

print(numpy.asarray([elem, cnt]))

'''
1 => false positive
0 => correct prediction
-1 => false negative
'''

tp = cnt[1]
fp = cnt[2]
fn = cnt[0]

precision = tp / (tp + fp)
recall = tp / (tp + fn)

F = 2 * precision * recall / (precision + recall)

print("\nF measure: " + str(F))


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                3) k-NN classifier
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

