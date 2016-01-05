import csv
import numpy


def csv_to_numpy(filename):
    file_reader = csv.reader(open(filename, 'r'))
    data = []
    next(file_reader)
    for row in file_reader:
        data.append(row)

    return numpy.array(data)

