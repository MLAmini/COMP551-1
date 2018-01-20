import pandas as pandas
import numpy as numpy
import matplotlib.pyplot as plt

trainingSet_1 = pandas.read_csv('./Datasets/Dataset_1_train.csv', header=None)
validSet_1 = pandas.read_csv('./Datasets/Dataset_1_valid.csv', header=None)
testSet_1 = pandas.read_csv('./Datasets/Dataset_1_test.csv', header=None)

testSet_1.drop([2], axis=1, inplace=True)

train1_x = trainingSet_1[0]
train1_y = trainingSet_1[1]
trainingSet_1.drop([1, 2], axis=1, inplace=True)

valid1_x = validSet_1[0]
valid1_y = validSet_1[1]
validSet_1.drop([1, 2], axis=1, inplace=True)


test1_x = testSet_1[0]
test1_y = testSet_1[1]



'''
Part 1
'''
#1) Mean Square Error

'''
20 degree polynomial
'''

N = 21;


#training set
train_matrix = trainingSet_1;
train_matrix.columns = [1]
for i in range(N):
    train_matrix[i] = pow(train_matrix[1], i)

train_matrix = train_matrix.sort_index(axis=1 ,ascending=True)

#validation set
valid_matrix = validSet_1;
valid_matrix.columns = [1]
for i in range(N):
    valid_matrix[i] = pow(valid_matrix[1], i)

valid_matrix = valid_matrix.sort_index(axis=1 ,ascending=True)


#weight function w* = (X^T X)^-1 X^T y

XTX_inv = numpy.linalg.pinv(numpy.matmul(train_matrix.T, train_matrix))
XT_y = numpy.matmul(train_matrix.T, train1_y)
weight = numpy.matmul(XTX_inv, XT_y)


'''
Mean Square Error computation
MSE = 1/N * sum((xw - y)^2)
'''


#TRAINING SET
wx_train = numpy.matmul(train_matrix, weight)

train_error = numpy.power(numpy.subtract(wx_train, train1_y), 2)
train_mse = train_error.mean()


print("\nTraining set MSE: ", train_mse)

#VALIDATION SET
wx_valid = numpy.matmul(valid_matrix, weight)

valid_error = numpy.power(numpy.subtract(wx_valid, valid1_y), 2)
valid_mse = valid_error.mean()

print("\nValidation set MSE: ", valid_mse)



plt.plot([1,2,3,4])
plt.ylabel('some numbers')
plt.show()










