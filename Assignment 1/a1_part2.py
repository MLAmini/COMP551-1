import pandas as pandas
import numpy as numpy
import matplotlib.pyplot as plt
import random

step = 1e-4
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


def linReg(train_input, train_output, valid_input, valid_output, step):
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

            print(w0, w1)

        train_mse.append(linMSE(train_input, train_output, w0, w1))
        valid_mse.append(linMSE(valid_input, valid_output, w0, w1))

        if (abs(w0 - w0_prev) < error) and (abs(w1 - w1_prev) < error):
            return train_mse, valid_mse, w0, w1


def linRegStep(train_input, train_output, test_input, test_output, steps):
    w0 = float(random.random() * 10)
    w1 = float(random.random() * 10)
    test_mse = []
    for i in range(len(steps)):
        print(steps[i])
        for j in range(1000):
            for k in range(len(train_input)):
                pred = w0 + train_input[k] * w1
                w0 = w0 - steps[i] * (pred - train_output[k])
                w1 = w1 - steps[i] * (pred - train_output[k]) * train_input[k]
        test_mse.append(linMSE(test_input, test_output, w0, w1))
    return test_mse


'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                                    Part 2
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# 1) Linear Regression
mse_train, mse_valid, w0, w1 = linReg(train2_x, train2_y, valid2_x, valid2_y, step)

print(mse_train, mse_valid)
print(len(mse_train))

epoch = [i for i in range(len(mse_train))]

plt.figure(1)
plt.plot(epoch[1:], mse_train[1:], 'b-')
plt.plot(epoch[1:], mse_valid[1:], 'r--')

plt.show()

# # 2) finding best step
steps = [50 * i * 1e-6 for i in range(1, 50)]
valid_mse_step = linRegStep(train2_x, train2_y, valid2_x, valid2_y, steps)

print(valid_mse_step)

plt.plot(steps, valid_mse_step, 'r--')
plt.show()

index = numpy.argmin(valid_mse_step)

print(index)
print(valid_mse_step[index])
print(steps[index])

step = steps[index]
mse_train, mse_test, w0, w1 = linReg(train2_x, train2_y, valid2_x, valid2_y, step)
epoch = [i for i in range(len(mse_test))]

plt.figure(1)
plt.plot(epoch[1:], mse_test[1:], 'r--')

plt.show()



# 3) print out the training curve
w0 = float(random.random() * 10)
w1 = float(random.random() * 10)

params = []

while True:
    w0_prev = w0
    w1_prev = w1
    for i in range(len(train2_x)):
        pred = w0 + train2_x[i] * w1
        w0 = w0 - step * (pred - train2_y[i])
        w1 = w1 - step * (pred - train2_y[i]) * train2_x[i]

    params.append([w0, w1])

    if (abs(w0 - w0_prev) < error) and (abs(w1 - w1_prev) < error):
        break

for i in range(len(params)):
    # index = int(i * len(params) / 5)
    index = i
    weight = params[index]
    pred = weight[1] * test2_x + weight[0]
    plt.figure(i + 1)
    plt.scatter(test2_x, test2_y)
    plt.plot(test2_x, pred, 'r-')

plt.show()