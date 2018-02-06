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
#
len0 = len(train0)
len1 = len(train1)


# drop the output column
train0 = train[train[20] == 0]
train0 = train0[train0.columns.difference([20])]

train1 = train[train[20] == 1]
train1 = train1[train1.columns.difference([20])]

test_output = test[20]
test.drop([20], axis=1, inplace=True)

train_output = train[20]
train.drop([20], axis=1, inplace=True)

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                2) LDA model using maximum likelihood approach
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# estimated max probability
print("\n\nPart 1 - LDA \n")
prob_0 = float(len0) / float(len0 + len1)
prob_1 = 1.0 - prob_0


mean_0 = numpy.array(train0.mean())
mean_1 = numpy.array(train1.mean())

diff_0 = numpy.array(train0 - mean_0)
diff_1 = numpy.array(train1 - mean_1)

coeff_0 = numpy.matmul(diff_0.T, diff_0)
coeff_1 = numpy.matmul(diff_1.T, diff_1)

coeff = (coeff_0 + coeff_1) / float(len0 + len1)

covm_0 = numpy.matmul(numpy.linalg.pinv(coeff), mean_0)
mcovm_0 = numpy.matmul(mean_0.T, covm_0)

covm_1 = numpy.matmul(numpy.linalg.pinv(coeff), mean_1)
mcovm_1 = numpy.matmul(mean_1.T, covm_1)

w0 = math.log(prob_0) - math.log(prob_1) - 1 / 2 * (mcovm_0 - mcovm_1)
w1 = numpy.matmul(numpy.linalg.pinv(coeff), mean_0 - mean_1)


print("w0: ", w0)
print("w1: " + str([i for i in w1]) + "\n")

pred = numpy.matmul(test, w1) + w0

# replace dec_bound to 0 and 1
pred[pred > 0] = 0
pred[pred < 0] = 1

error_matrix = pred - test_output

elem, cnt = numpy.unique(error_matrix, return_counts=True)

print(numpy.asarray([elem, cnt]))

'''
1 => false positive
0 => correct prediction
-1 => false negative
'''

fp = cnt[2]
fn = cnt[0]


tp = numpy.sum(pred.dot(test_output))
print("True positive: " + str(tp))

pred = pred - 1
test_output_temp = test_output - 1
tn = numpy.sum(pred.dot(test_output_temp))
print("True negative: " + str(tn))

precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tn + tp) / (tn + tp + fn + fp)

F = 2 * precision * recall / (precision + recall)

print("\nF measure: " + str(F))
print("Accuracy: " + str(accuracy))
print("Precision: " + str(precision))
print("Recall: " + str(recall))



'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                3) k-NN classifier
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
print("\n\nPart 2 - K NN\n")


res_step = []
accuracy_step = []


for k in range(1, 50):
    # metrics = [tp, tn, fp, fn]
    metrics = [0, 0, 0, 0]
    for i in range(len(test)):
        dist = abs(train.sub(numpy.array(numpy.array(test.loc[[i], :])[0])))
        dist = numpy.array(numpy.power(dist, 2).sum(axis=1))

        # select the k training examples closest to the test example
        ind = numpy.argpartition(dist, k)[:k]

        # predict value of test output using the average of the k training outputs
        dec = 1 if (numpy.array([train_output[i] for i in ind]).mean() > 0.5) else 0

        #update metrics
        if dec == test_output[i] == 1:
            metrics[0] += 1
        elif dec == test_output[i] == 0:
            metrics[1] += 1
        elif dec != test_output[i] == 1:
            metrics[2] += 1
        else:
            metrics[3] += 1

    res_step.append(metrics)
    accuracy_step.append((metrics[0] + metrics[1]) / numpy.sum(metrics))
print(res_step)
print(accuracy_step)

print("\nBest k = " + str(accuracy_step.index(max(accuracy_step)) + 1))

metrics = res_step[accuracy_step.index(max(accuracy_step))]
tp = metrics[0]
tn = metrics[1]
fp = metrics[2]
fn = metrics[3]

precision = tp / (tp + fp)
recall = tp / (tp + fn)
accuracy = (tn + tp) / (tn + tp + fn + fp)

F = 2 * precision * recall / (precision + recall)

print("\nAccuracy = " + str(max(accuracy_step)))
print("Precision = " + str(precision))
print("Recall = " + str(recall))
print("F - measure = " + str(F))


