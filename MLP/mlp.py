from PerceptronLayer import *
from MLP import *
import os.path

def fn(x):
    return 1/(1+np.exp(-x))
def dfn(x):
    y = fn(x)
    return y * (1 - y)

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

def create_mlp_clasificator(filename, otherfilename, learning_rate, mlp_dimentions, max_epoch):
    (inputs_class_1, expected_values_class_1) = get_inputs(filename, [1,0])
    (inputs_class_2, expected_values_class_2) = get_inputs(otherfilename, [0,1])
    inputs = inputs_class_1 + inputs_class_2
    expected_values = expected_values_class_1 + expected_values_class_2
    try:
        N = len(inputs[0])
        etha = learning_rate
        mlp = MLP(etha, fn, dfn, N, mlp_dimentions)
        errors = mlp.train(inputs, expected_values, max_epoch)
        return errors, mlp, inputs, expected_values
    except:
        raise Exception("Error in clasification")

def compare_learing_rates(filename, otherfilename, learning_rates, mlp_dimentions, max_epoch):
    for i in range(len(learning_rates)):
        errors, mlp,_,_ = create_mlp_clasificator(filename, otherfilename, learning_rates[i], mlp_dimentions, max_epoch)
        for layer in mlp._network:
            print(layer._weight)
        plot_points(errors)

def plot_points():

    
def main():
    earth = 'datos/EarthSpace - EarthSpace.csv'
    med = 'datos/MedSci - MedSci.csv'
    lf = 'datos/LifeSci - LifeSci.csv'
    agr = 'datos/Agri - Agri.csv'
    learning_rates = [ 0.1 ]
    dimentions = (4,2)

    compare_learing_rates(earth, med, learning_rates, dimentions, 1000)
    compare_learing_rates(lf, agr, learning_rates, dimentions, 1000)

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


