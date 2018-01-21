import pandas as pandas
import numpy as numpy
import matplotlib.pyplot as plt
import random

step = 1e-3
error = 1e-4

trainingSet_2 = pandas.read_csv('./Datasets/Dataset_2_train.csv', header=None)
validSet_2 = pandas.read_csv('./Datasets/Dataset_2_valid.csv', header=None)
testSet_2 = pandas.read_csv('./Datasets/Dataset_2_test.csv', header=None)


train2_x = trainingSet_2[0]
train2_y = trainingSet_2[1]
trainingSet_2.drop([2], axis=1, inplace=True)

valid2_x = validSet_2[0]
valid2_y = validSet_2[1]
validSet_2.drop([2], axis=1, inplace=True)

test2_x = testSet_2[0]
test2_y = testSet_2[1]
testSet_2.drop([2], axis=1, inplace=True)

def linMSE(input, output, w0, w1):
    predict = input * w0 + w1
    return numpy.power(numpy.subtract(predict, output), 2).mean()


def linReg(train_input, train_output, valid_input, valid_output):
    w0 = float(random.random() * 10)
    w1 = float(random.random() * 10)
    train_mse = []
    valid_mse = []
    while True:
        w0_prev = w0
        w1_prev = w1
        for i in range(len(train_input)):
            pred = w0 + train_input[i] * w1
            w0 = w0 - step * (pred - train_output[i])
            w1 = w1 - step * (pred - train_output[i]) * train_input[i]

        train_mse.append(linMSE(train_input, train_output, w0, w1))
        valid_mse.append(linMSE(valid_input, valid_output, w0, w1))

        if (abs(w0 - w0_prev) < error) and (abs(w1 - w1_prev) < error):
            return train_mse, valid_mse, w0, w1


def linRegStep(train_input, train_output, valid_input, valid_output, steps):
    w0 = float(random.random() * 10)
    w1 = float(random.random() * 10)
    valid_mse = []
    for i in range(len(steps)):
        for j in range(1000):
            for k in range(len(train_input)):
                pred = w0 + train_input[k] * w1
                w0 = w0 - steps[i] * (pred - train_output[k])
                w1 = w1 - steps[i] * (pred - train_output[k]) * train_input[k]

        valid_mse.append(linMSE(valid_input, valid_output, w0, w1))
    return valid_mse



'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                                    Part 2
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

#1) Linear Regression
# linReg  = linReg(train2_x, train2_y, valid2_x, valid2_y)
# mse_train = linReg[0]
# mse_valid = linReg[1]
# w0 = linReg[2]
# w1 = linReg[3]
#
# print(mse_train, mse_valid)
#
# epoch = [i for i in range(len(mse_train))]

# plt.figure(1)
# plt.plot(epoch, mse_train, 'b-')
# plt.plot(epoch, mse_valid, 'r--')
#
# plt.show()

#2) finding best step
steps = [i * 1e-6 for i in range(int(1e4))]
mse_valid_step = linRegStep(train2_x, train2_y, valid2_x, valid2_y, steps)

print(mse_valid_step)

plt.plot(steps, mse_valid_step, 'r--')
plt.show()






