"""
Aproximador universal usando como base funciones de base radial.
Se reciben como parametros:
- inputs: matriz de estimulos de entrenamiento
- expected: Vector de valores esperados por cada estimulo
- function: funcion de base radial a aplicar a cada estimulo para formar las neuronas
- factor_regulation: Es el parametro de regularizacion

Se pueden acceder a los metodos:
- interpolate(x) con x un vector de entrada al cual interpolar
"""
import numpy as np
class UniAprox:
    """
    Se crea una neurona por dato de entrada y luego se resuelve el sistema de ecuaciones
    asociado para determinar los pesos sinapticos de cada neurona
    """
    def __init__(self, inputs, expected, function, factor_regulation = 0) -> None:
        ( num_neuron, num_dimention ) = inputs.shape
        self.num_neuron = num_neuron
        self.num_dimention = num_dimention
        self.inputs = inputs
        self.factor = factor_regulation
        self.function = function
        G = np.zeros(( num_neuron, num_neuron ))
        for i in range(num_neuron):
            for j in range(num_neuron):
                G[i][j] = function(self.distance(inputs[i], inputs[j]))
        self.G = G

        self.W = None
        try:
            self.W = np.linalg.solve( G + self.factor*np.identity(num_neuron), expected )
        except:
            raise "Error al resolver el sistema lineal"

    def get_weights(self):
        return self.W
    
    def distance(self, np_vector1, np_vector2) -> float:
        return np.linalg.norm(np_vector1 - np_vector2)
    
    def interpolate( self, input ):
        phi = np.zeros((self.num_neuron))
        for i in range(self.num_neuron):
            phi[i] = self.function(self.distance(input, self.inputs[i]))
        return np.dot( phi, self.W )[0]
    

        
