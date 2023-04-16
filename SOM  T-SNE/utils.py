import csv
import numpy as np

"""
Se obtiene el conjunto de datos en el archivo proporcionado
"""
def get_training_set( filename ):
    with open(filename, 'r', encoding='utf-8') as f:
        csvreader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        lines = []
        for row in csvreader:
            lines.append(row)
        lines = np.array(lines)
        inputs = lines[:, [0]]
        expected = lines[:, [1]]
        stddev = stat.stdev(lines[:, 0])
        return (inputs, expected, stddev)
    
def get_data( filename ):
    with open(filename, 'r', encoding='utf-8') as f:
        csvreader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        lines = []
        for row in csvreader:
            lines.append(row)
        lines = np.array(lines)
        return lines
