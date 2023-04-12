"""
Programa cliente de pruebas sobre las clases UniAprox y RBF
se inicializan dichas clases y luego se aproxima la curva dados los puntos
en el archivo Spectra 100. Ademas se mide la eficacia de la aproximacion
midiendo el error cuadratico medio cometido entre la curva estimada y la curva real
que se encuentra en el archivo SpectraReal
"""

from UniAprox import *
from RBF import *
import statistics as stat
import csv
import matplotlib.pyplot as plt

dev = -1
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

"""
Funcion Gaussiana
"""
def gaussian(x):
    return np.exp( -1/(2*dev**2) * x )

"""
Se calcula el error cuadratico medio cometido entre la curva estimada y la curva real
que se encuentra en el archivo SpectraReal
"""
def get_metric(obj):
    ( x_real, y_real, _ ) = get_training_set("../datos/SpectraReal - SpectraReal.csv")
    error = 0
    for xi, yi in zip ( x_real, y_real):
        y = obj.interpolate(xi)
        error += 0.5*(yi[0]-y)**2
    print(error)

"""
Se grafican los datos recibidos como argumentos
"""
def plot(inputs, expected, obj, title, xlabel, ylabel, centroids=False):
    x = list(inputs.T[0])
    y = []
    for input in x:
        y.append( obj.interpolate( [ input ] ) )
    plt.plot(x,y)
    if centroids:
        plt.plot(obj.center_x, obj.center_y, 'bo')
    plt.plot(inputs.T[0], expected, 'ro')
    plt.axis([0.09, 5, 0, 1.3] )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    print("Graficando Curva interpoladora y datos")
    plt.show()

"""
Funcion de pruebas para el Aproximador universal
"""
def uniAprox(factor=0):
    global dev
    ( inputs, expected, dev )   = get_training_set( "../datos/Spectra100 - Spectra100.csv" )
    uni = UniAprox( inputs, expected, gaussian, factor )
    plot(inputs, expected, uni, f"Universal Aproximator lambda = {factor}", "", "")
    get_metric(uni)

"""
Funcion de pruebas para el RBF
"""
def rbfAprox( neurons, factor=0, plotter=True):
    global dev
    ( inputs, expected, dev )   = get_training_set( "../datos/Spectra100 - Spectra100.csv" )
    data = np.concatenate((inputs, expected), axis=1)
    rbf = RBF( neurons, data, gaussian, factor)
    if(plotter):
        plot(inputs, expected, rbf, f"RBF con {neurons} neuronas lambda = {factor}", "", "", True)
    get_metric(rbf)
    return rbf.inertia
"""
Methodo del Codo (Elbow Method) para calcular la menor cantidad de clusters
"""
def draw_elbow_method():
    # Calculando mejor cantidad de Clusters (Elbow Method)
    x = list(range(5,50))
    y = []
    for xi in x:
        r = rbfAprox(xi, plotter=False)
        y.append(r)
    plt.plot(x,y)
    plt.title("Inercia por Cantidad de Clusters")
    plt.xlabel("Num Clusters")
    plt.ylabel("Inercia")
    plt.show()

"""
Programa Principal
"""
def main():
    print("Programa Cliente Iniciado")
    # Aproximador universal
    lambdas = [ 0, 0.001, 0.01, 0.1, 1]
    for lamb in lambdas:
        uniAprox(lamb)
    
    #RBF
    draw_elbow_method() # Para observar la mejor cantidad de neuronas

    # Se busca un buen lambda
    lambdas = [ 0, 0.001, 0.01, 0.1]
    neurons = 30
    for lamb in lambdas:
        rbfAprox(neurons, lamb)
    
    

if __name__ == '__main__':
    main()