from UniAprox import *
from RBF import *
import statistics as stat
import csv
import matplotlib.pyplot as plt

dev = -1
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
    
def gaussian(x):
    return np.exp( -1/(2*dev**2) * x )

def plot(inputs, expected, obj):
    # x = list(inputs.T[0]) + list(np.arange( 0, 6, 0.1))
    x = list(inputs.T[0])
    # x.sort()
    y = []
    for input in x:
        y.append( obj.interpolate( [ input ] ) )
    print(y, expected)
    plt.plot(x,y)
    plt.plot(inputs.T[0], expected, 'ro')
    plt.axis([0.09, 5, 0, 1.3] )
    print("Graficando Curva interpoladora y datos")
    plt.show()

def uniAprox():
    global dev
    ( inputs, expected, dev )   = get_training_set( "../datos/Spectra100 - Spectra100.csv" )
    print(expected)
    uni = UniAprox( inputs, expected, gaussian, 0.03 )
    plot(inputs, expected, uni)

def rbfAprox():
    global dev
    ( inputs, expected, dev )   = get_training_set( "../datos/Spectra100 - Spectra100.csv" )
    data = np.concatenate((inputs, expected), axis=1)
    rbf = RBF( 20, data, gaussian, 0)
    plot(inputs, expected, rbf)


def main():
    print("Programa Cliente Iniciado")
    uniAprox()
    rbfAprox()

    
    




if __name__ == '__main__':
    main()