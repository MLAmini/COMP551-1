import pandas as pandas
import numpy as numpy
import matplotlib.pyplot as plt
import random

trainingSet_1 = pandas.read_csv('./Datasets/Dataset_1_train.csv', header=None)
validSet_1 = pandas.read_csv('./Datasets/Dataset_1_valid.csv', header=None)
testSet_1 = pandas.read_csv('./Datasets/Dataset_1_test.csv', header=None)

trainingSet_1.sort_values(by=[0], inplace=True)
validSet_1.sort_values(by=[0], inplace=True)

train1_x = trainingSet_1[0]
train1_y = trainingSet_1[1]
trainingSet_1.drop([1, 2], axis=1, inplace=True)

valid1_x = validSet_1[0]
valid1_y = validSet_1[1]
validSet_1.drop([1, 2], axis=1, inplace=True)


test1_x = testSet_1[0]
test1_y = testSet_1[1]
testSet_1.drop([1, 2], axis=1, inplace=True)


def weightCalc(matrix, output):
    XTX_inv = numpy.linalg.pinv(numpy.matmul(matrix.T, matrix))
    XT_y = numpy.matmul(matrix.T, output)
    return numpy.matmul(XTX_inv, XT_y)


def weightCalc_reg(matrix, l, output):
    XTX_inv = numpy.linalg.pinv(numpy.matmul(matrix.T, matrix) + l * numpy.identity(21))
    XT_y = numpy.matmul(matrix.T, output)
    return numpy.matmul(XTX_inv, XT_y)


def mseCalc(matrix, weight, output):
    predict  = numpy.matmul(matrix, weight)
    train_error = numpy.power(numpy.subtract(predict, output), 2)
    return train_error.mean()

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                                    Part 1
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#1) Mean Square Error

'''
20 degree polynomial
'''

N = 21;

#training set
train_matrix = trainingSet_1;
train_matrix.columns = [1]
for i in range(N):
    train_matrix[i] = pow(train_matrix[1], numpy.float64(i))

train_matrix = train_matrix.sort_index(axis=1 ,ascending=True)

#validation set
valid_matrix = validSet_1;
valid_matrix.columns = [1]
for i in range(N):
    valid_matrix[i] = pow(valid_matrix[1], numpy.float64(i))

valid_matrix = valid_matrix.sort_index(axis=1 ,ascending=True)


#test set
test_matrix = testSet_1;
test_matrix.columns = [1]
for i in range(N):
    test_matrix[i] = pow(test_matrix[1], numpy.float64(i))

test_matrix = test_matrix.sort_index(axis=1 ,ascending=True)



#weight function w* = (X^T X)^-1 X^T y

weight = weightCalc(train_matrix, train1_y)

'''
Mean Square Error computation
MSE = 1/N * sum((xw - y)^2)
'''


#TRAINING SET
predict_train = numpy.matmul(train_matrix, weight)

train_error = numpy.power(numpy.subtract(predict_train, train1_y), 2)
train_mse = train_error.mean()


print("\nTraining set MSE: ", train_mse)

#VALIDATION SET
predict_valid = numpy.matmul(valid_matrix, weight)

valid_error = numpy.power(numpy.subtract(predict_valid, valid1_y), 2)
valid_mse = valid_error.mean()

print("\nValidation set MSE: ", valid_mse)

#Test SET
predict_test = numpy.matmul(test_matrix, weight)

test_error = numpy.power(numpy.subtract(predict_test, test1_y), 2)
test_mse = test_error.mean()

print("\nTest set MSE: ", test_mse)

#2) add L2 regularization

train_mse_reg = []
valid_mse_reg = []
test_mse_reg = []

min_valError = float('inf')
lambda_minError = 0

lambdas = [0.001 * i for i in range(1000)]
for i in lambdas:
    weight_reg = weightCalc_reg(train_matrix, i, train1_y)

    #TRAINING SET
    predict_train = numpy.matmul(train_matrix, weight_reg)

    train_error = numpy.power(numpy.subtract(predict_train, train1_y), 2)
    train_mse = train_error.mean()

    train_mse_reg.append(train_mse)

    #VALIDATION SET
    predict_valid = numpy.matmul(valid_matrix, weight_reg)

    valid_error = numpy.power(numpy.subtract(predict_valid, valid1_y), 2)
    valid_mse = valid_error.mean()

    if(valid_mse < min_valError):
        min_valError = valid_mse
        lambda_minError = i

    valid_mse_reg.append(valid_mse)

    #TEST SET
    predict_test = numpy.matmul(test_matrix, weight_reg)

    test_error = numpy.power(numpy.subtract(predict_test, test1_y), 2)
    test_mse = test_error.mean()

    test_mse_reg.append(test_mse)

#visualize part 1
plt.figure(1)
plt.subplot(3, 1, 1)
plt.scatter(train1_x, predict_train)
plt.plot(train1_x, train1_y, 'r--')

plt.subplot(3, 1, 2)
plt.scatter(valid1_x, predict_valid)
plt.plot(train1_x, train1_y, 'r--')


plt.subplot(3, 1, 3)
plt.scatter(test1_x, predict_test)
plt.plot(train1_x, train1_y, 'r--')


#visualize part 2
plt.figure(2)
plt.plot(lambdas[1:], train_mse_reg[1:], 'b-', label="train")
plt.plot(lambdas[1:], valid_mse_reg[1:], 'r--', label="validation")
plt.plot(lambdas[1:], test_mse_reg[1:], 'g-', label="test")

plt.scatter([lambda_minError], [valid_mse_reg[int(lambda_minError*1000)]], color='r')

# Place a legend to the right of this smaller subplot.
plt.legend(bbox_to_anchor=(1, 1),
           bbox_transform=plt.gcf().transFigure, borderaxespad=0.)

plt.show()

