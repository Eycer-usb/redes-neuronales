from PerceptronLayer import *
from MLP import *
import os.path

def fn(x):
    return 1/(1+np.exp(-x))
def dfn(x):
    y = fn(x)
    return y * (1 - y)

def get_inputs( file_name, expected_value):
    with open(f'{os.path.dirname(__file__)}/../{file_name}', "r", encoding='utf-8') as f:
        lines = f.readlines()
        inputs = list(map( lambda line: 
                list( map( lambda data: float(data), line.split(',')) ),
                lines))
        expected_values = (np.zeros((len(lines), 1)) + expected_value).tolist()
        return (inputs, expected_values) 

def create_mlp_clasificator(filename, otherfilename, learning_rate, mlp_dimentions, max_epoch):
    (inputs_class_1, expected_values_class_1) = get_inputs(filename, 0)
    (inputs_class_2, expected_values_class_2) = get_inputs(otherfilename, 1)
    inputs = inputs_class_1 + inputs_class_2
    expected_values = expected_values_class_1 + expected_values_class_2
    try:
        N = len(inputs[0])
        etha = learning_rate
        mlp = MLP(etha, fn, dfn, N, mlp_dimentions)
        errors = mlp.train(inputs, expected_values, max_epoch)
        return errors
    except:
        raise Exception("Error in clasification")

def compare_learing_rates(filename, otherfilename, learning_rates, mlp_dimentions, max_epoch):
    for i in range(len(learning_rates)):
        errors = create_mlp_clasificator(filename, otherfilename, learning_rates, mlp_dimentions, max_epoch)
        print(errors)

def main():
    earth = 'datos/EarthSpace - EarthSpace.csv'
    med = 'datos/MedSci - MedSci.csv'
    lf = 'datos/LifeSci - LifeSci.csv'
    agr = 'datos/Agri - Agri.csv'
    # learning_rates = [ 0.001, 0.01, 0.1, 1 ]
    learning_rates = [0.1]
    dimentions = (4,3,2,1)

    compare_learing_rates(earth, med, learning_rates, dimentions, 5)
    compare_learing_rates(lf, agr, learning_rates, dimentions, 5)
   

if __name__ == '__main__':
    main()