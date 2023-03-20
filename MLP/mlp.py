"""
Client Program for MLP class
"""

from PerceptronLayer import *
from MLP import *
import os.path
import matplotlib.pyplot as plt

"""
Activation non-linear Function for the MLP
Will use the de Sigmoid function
"""
def fn(x):
    return 1/(1+np.exp(-x))

"""
First Derivative of Sigmoid activation function
"""
def dfn(x):
    y = fn(x)
    return y * (1 - y)

"""
Argumuments:
file_name: File name where data is storanged
expected_value: expected value for class

It will read the input file and return the stimuls list and expected answers list
"""
def get_inputs( file_name, expected_value ):
    with open(f'{os.path.dirname(__file__)}/../{file_name}', "r", encoding='utf-8') as f:
        lines = f.readlines()
        inputs = list(map( lambda line: 
                list( map( lambda data: float(data), line.split(',')) ),
                lines))
        expected_values = []
        for _ in range(len(lines)):
            expected_values.append(expected_value)
        return (inputs, expected_values) 

"""
Arguments:
filename: First class filename
otherfilename: Second class filename
learnig_rate: Learning rate to the entire MLP network
mlp_dimentions: Tuple with the structure for MLP. For example (4, 2) means 4 neurons in first layer and 2 at out layer
max_epoch: Number of epoch to train the network

A MLP network is trained with input data
"""
def create_mlp_clasificator(filename, otherfilename, learning_rate, mlp_dimentions, max_epoch, alpha):
    (inputs_class_1, expected_values_class_1) = get_inputs(filename, [1,0])
    (inputs_class_2, expected_values_class_2) = get_inputs(otherfilename, [0,1])
    inputs = inputs_class_1 + inputs_class_2
    expected_values = expected_values_class_1 + expected_values_class_2
    try:
        N = len(inputs[0])
        etha = learning_rate
        mlp = MLP(etha, fn, dfn, N, mlp_dimentions, alpha)
        errors = mlp.train(inputs, expected_values, max_epoch)
        return errors, mlp, inputs, expected_values
    except:
        raise Exception("Error in clasification")

"""
A set of learning rates is used to compare performance with the same data
and a graphic is plotted with comparative
"""
def compare_learing_rates(filename, otherfilename, learning_rates, mlp_dimentions, max_epoch, alpha):
    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    for i in range(len(learning_rates)):
        errors, mlp,_,_ = create_mlp_clasificator(filename, otherfilename, learning_rates[i], mlp_dimentions, max_epoch, alpha)
        print(errors[-1])
        plot_error(max_epoch, errors, learning_rates[i], colors.pop())
    plt.ylabel('Medium Square Error')
    plt.xlabel('Epoch number')
    plt.legend()
    plt.show()

"""
In the actual graphic plot error vs epoch
"""
def plot_error(max_epoch, errors, learning_rate, color):
    x = list(range(max_epoch))
    y = errors
    plt.plot(x,y, color, label = f"Etha: {learning_rate}")

"""
Start Point
"""
def main():
    earth = 'datos/EarthSpace - EarthSpace.csv'
    med = 'datos/MedSci - MedSci.csv'
    lf = 'datos/LifeSci - LifeSci.csv'
    agr = 'datos/Agri - Agri.csv'
    learning_rates = [ 0.01 ]
    dimentions = (4,2)

    # compare_learing_rates(earth, med, learning_rates, dimentions, 600, 0.3)
    compare_learing_rates(lf, agr, learning_rates, dimentions, 100, 0.2)

"""
Testing Function
"""
def main2():
    inputs = [[1,0], [0,1]]
    expected = [[1], [0]]
    mlp = MLP(10,fn,dfn,2,(2,1))
    prt = mlp.train(inputs, expected, 1000)
    print(mlp.activate(inputs[0]))
    print(mlp.activate(inputs[1]))
    print(prt[-1])


if __name__ == '__main__':
    main()


