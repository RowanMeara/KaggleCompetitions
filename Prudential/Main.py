from getdata import csv_to_numpy
import numpy
import sklearn

if __name__ == "__main__":
    data = csv_to_numpy('train.csv')
    print(data[0])
