import pandas
import numpy
import matplotlib.pyplot as plt
import random

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

'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
                                                    Part 3
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# data has 1993 rows and 128 columns (one of which is the name of the city)
dataset = pandas.read_csv('crime_dataset.csv', header=None)
dataset.drop([0, 1, 2, 3, 4], axis=1, inplace=True)
dataset = dataset.replace('?', -999999).astype(numpy.float64).replace(-999999, numpy.nan)
dataset.fillna(dataset.mean(), inplace=True)
#
# '''
# The generated train / test data sets are saved in separate csv files as requested.
# In order to maintain the same data set for the report, I will not regenerate the data sets.
# This part of the code will be commented out
# '''
#
# generate 5 separate 80-20 data

for i in range(1, 6):
    temp_data = dataset
    # generate 20% random rows to be put in the training set
    msk = numpy.random.rand(len(temp_data)) <= 0.8
    train = temp_data[msk]
    test = temp_data[~msk]
    train.to_csv('CandC-train' + str(i) + '.csv')
    test.to_csv('CandC-test' + str(i) + '.csv')

step = 1e-3
mse = []

for i in range(1, 6):
    ds, viol = parsedata('CandC-train' + str(i) + '.csv')
    test, test_output = parsedata('CandC-test' + str(i) + '.csv')
    weight = weightCalc(ds, viol)
    numpy.savetxt('3_2_weights' + str(i) + 'noreg.csv', weight, delimiter=',')

    prediction = numpy.matmul(test, weight)
    mse.append(mseCalc(prediction, test_output))

print("without regularization: " + str(numpy.mean(mse)))

error = []
lambdas = [0.02 * i for i in range(500)]

avg_mse = []
ds = []
viol = []
test = []
test_output = []

for i in range(1, 6):
    ds_temp, viol_temp = parsedata('CandC-train' + str(i) + '.csv')
    test_temp, test_output_temp = parsedata('CandC-test' + str(i) + '.csv')
    ds.append(ds_temp)
    viol.append(viol_temp)
    test.append(test_temp)
    test_output.append(test_output_temp)


for l in lambdas:
    mse = []
    mse_list = []

    for i in range(5):
        weight = weightCalc_reg(ds[i], l, viol[i])
        prediction = numpy.matmul(test[i], weight)
        mse_list = mseCalc(prediction, test_output[i])
        mse.append(mse_list)
    avg_mse.append(numpy.mean(mse))
    error.append(mse)

index = numpy.argmin(avg_mse)
print("the best regularization constant lambda is " + str(lambdas[index]))
print("MSE = " + str(error[index]))
print("the average MSE = " + str(numpy.mean(error[index])))

# save the trained parameters using the correct step
weight_reg = []
for i in range(5):
    weight = weightCalc_reg(ds[i], lambdas[index], viol[i])
    numpy.savetxt('3_3_weights' + str(i) + 'reg.csv', weight, delimiter=',')
    weight_reg.append(weight)

# 3_3) feature selection

weights = pandas.DataFrame()
for i in range(1, 6):
    weights[i] = weight_reg[i-1]

weights = numpy.array(pandas.DataFrame.mean(weights, axis=1))


mse_feature = []
mse_feature_ave = []

ind = numpy.argpartition(weights, -44)[-44:]

array = [i for i in [x for x in range(len(weights)) if x not in ind]]
mse = []
for i in range(5):
    ds[i].drop(array, axis=1, inplace=True)
    test[i].drop(array, axis=1, inplace=True)
    weight = weightCalc(ds[i], viol[i])
    prediction = numpy.matmul(test[i], weight)
    mse.append(mseCalc(prediction, test_output[i]))

mse_feature.append(mse)
mse_feature_ave.append(numpy.mean(mse))

print('5-fold cross-validation error with feature selection:' + str(mse))
print('overall average mse with feature selection:' + str(numpy.mean(mse)))

