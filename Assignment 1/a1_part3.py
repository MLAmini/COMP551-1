import pandas
import numpy
import random

# def linReg()

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                                    Part 3
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# data has 1993 rows and 128 columns (one of which is the name of the city)
dataset = pandas.read_csv('crime_dataset.csv', header=None)
dataset.drop([0, 1, 2, 3, 4], axis=1, inplace=True)
dataset = dataset.replace('?', -999999).astype(numpy.float64).replace(-999999, numpy.nan)
dataset.fillna(dataset.mean(), inplace=True)

'''
The generated train / test data sets are saved in separate csv files as requested.
In order to maintain the same data set for the report, I will not regenerate the data sets.
This part of the code will be commented out
'''

# generate 5 separate 80-20 data
for i in range(1, 6):
    temp_data = dataset
    # generate 20% random rows to be put in the training set
    msk = numpy.random.rand(len(temp_data)) < 0.8
    train = temp_data[msk]
    test = temp_data[~msk]
    train.to_csv('CandC-train' + str(i) + '.csv')
    test.to_csv('CandC-test' + str(i) + '.csv')

step = 1e-3
mse = []

def weightCalc(matrix, output):
    XTX_inv = numpy.linalg.pinv(numpy.matmul(matrix.T, matrix))
    XT_y = numpy.matmul(matrix.T, output)
    return numpy.matmul(XTX_inv, XT_y)


def weightCalc_reg(matrix, l, output):
    XTX_inv = numpy.linalg.pinv(numpy.matmul(matrix.T, matrix) + l * numpy.identity(len(matrix.columns)))
    XT_y = numpy.matmul(matrix.T, output)
    return numpy.matmul(XTX_inv, XT_y)


def mseCalc(predict, output):
    train_error = numpy.power(numpy.subtract(predict, output), 2)
    return train_error.mean()


def parsedata(filename):
    ds = pandas.read_csv(filename, header=None)
    ds.drop([0], axis=0, inplace=True)
    output = ds[len(ds.columns) - 1]
    ds.drop([0, len(ds.columns) - 1], axis=1, inplace=True)
    ds[0] = 1
    ds = ds.sort_index(axis=1, ascending=True)
    return ds, output


for i in range(1, 6):
    ds, viol = parsedata('CandC-train' + str(i) + '.csv')
    weight = weightCalc(ds, viol)

    test, test_output = parsedata('CandC-test' + str(i) + '.csv')
    prediction = numpy.matmul(test, weight)
    mse.append(mseCalc(prediction, test_output))

print("without regularization: " + str(numpy.mean(mse)))

error = []
lambdas = [0.002 * i for i in range(500)]

for l in lambdas:
    err = []
    for i in range(1, 6):
        ds, viol = parsedata('CandC-train' + str(i) + '.csv')
        test, test_output = parsedata('CandC-test' + str(i) + '.csv')
        weight = weightCalc_reg(ds, l, viol)
        prediction = numpy.matmul(test, weight)
        err.append(mseCalc(prediction, test_output))
    error.append(numpy.mean(err))

index = numpy.argmin(error)
print(numpy.argmin(error))
print(lambdas[index])
print(error[index])





# for i in range(1, 6):
#     ds = pandas.read_csv('CandC-train' + str(i) + '.csv', header=None)
#     ds.drop([0], axis=0, inplace=True)
#     viol = ds[len(ds.columns) - 1]
#
#     ds.drop([0, len(ds.columns) - 1], axis=1, inplace=True)
#     ds[0] = 1
#     ds = ds.sort_index(axis=1, ascending=True)
#     param = [random.random() * 10 for i in range(len(ds.columns))]
#     param_prev = [0 for i in range(len(ds.columns))]
#     ds = ds.T
#
#     while numpy.linalg.norm(numpy.subtract(param, param_prev)) > 1e-3:
#         param_prev = param
#         for j in range(1, len(ds) + 1):
#             scalar = step * (ds[j].dot(param) - viol[j])
#             param = numpy.subtract(param, scalar * ds[j])
#
#     test = pandas.read_csv('CandC-test' + str(i) + '.csv', header=None)
#     test.drop([0], axis=0, inplace=True)
#     test_output = test[len(test.columns) - 1]
#
#     test.drop([0, len(test.columns) - 1], axis=1, inplace=True)
#     test[0] = 1
#     test = test.sort_index(axis=1, ascending=True)
#
#     # test = test.T
#
#     predict = numpy.matmul(test, param)
#     train_error = numpy.power(numpy.subtract(predict, test_output), 2)
#     mse.append(train_error.mean())
#
# print(numpy.mean(mse))

# ds = pandas.read_csv('CandC-train' + str(1) + '.csv', header=None)
# ds.drop([0], axis=0, inplace=True)
# viol = ds[len(ds.columns) - 1]
#
# ds.drop([0, len(ds.columns) - 1], axis=1, inplace=True)
# ds[0] = 1
# ds = ds.sort_index(axis=1, ascending=True)
# param = [random.random() * 10 for i in range(len(ds.columns))]
# param_prev = [0 for i in range(len(ds.columns))]
# ds = ds.T
#
# while numpy.linalg.norm(numpy.subtract(param, param_prev)) > 1e-3:
#     param_prev = param
#     for j in range(1, len(ds) + 1):
#         scalar = step * (ds[j].dot(param) - viol[j])
#         param = numpy.subtract(param, scalar * ds[j])
#     print(numpy.linalg.norm(numpy.subtract(param, param_prev)))
#
# test = pandas.read_csv('CandC-test' + str(1) + '.csv', header=None)
# test.drop([0], axis=0, inplace=True)
# test_output = test[len(test.columns) - 1]
#
# test.drop([0, len(test.columns) - 1], axis=1, inplace=True)
# test[0] = 1
# test = test.sort_index(axis=1, ascending=True)
#
# # test = test.T
#
# predict = numpy.matmul(test, param)
# train_error = numpy.power(numpy.subtract(predict, test_output), 2)
# mse.append(train_error.mean())
#
# print(mse)




