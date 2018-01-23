import pandas as pandas
import numpy
import csv


x = {"0": [1, 1, 1, 0, 1, 1], "1": [2, 2, 2, 3, 0, 0]}
x = pandas.DataFrame(data=x)
x.astype(numpy.float)
x.replace(0, numpy.nan, inplace=True)
x.fillna(x.mean(), inplace=True)
print(x.mean())
print(x)


dataset = pandas.read_csv('crime_dataset.csv', header=None)
city = dataset[3]
dataset.drop([3], axis=1, inplace=True)
dataset = dataset.replace('?', -999999).astype(numpy.float64).replace(-999999, numpy.nan)

dataset.fillna(dataset.mean(), inplace=True)

dataset[len(dataset.columns) + 1] = city
print(dataset)

