"""
Implementacion del algoritmo del LMS para los 
dispositivos Adaline mediante el metodo de actualizacion en linea
o estocastica
"""

import numpy as np

class LMSStochastic:
    """
    Los pesos y los costos son arreglos que contienen los costos o error
    cuadratico medio en cada iteracion
    """
    def __init__(self):
        self.costos = []
        self.pesos = []

    """
    Funcion de entrenamiento del Adaline

    ARGUMENTOS:
    estimulos: es la lista de estimulos recibidos (sin coordenada asociada al sesgo)
    respuestas: lista de respuestas deseadas correspondientes a los estimulos
    etha: tasa de aprendizaje del LMS
    epocas: numero maximo de epocas a iterar

    """
    def entrenar( self, estimulos, respuestas, etha, epocas ):

        (nro_estimulos, nro_entradas)  = estimulos.shape
        sesgo = np.array([1])
        self.epocas = epocas
        self.etha = etha
        if self.pesos == []:
            self.iniciarPesos(nro_entradas, sesgo, -0.05, 0.05)
        estimulos_ext = np.concatenate((estimulos, np.atleast_2d(np.ones(nro_estimulos)).T), axis=1)

        for i in range(self.epocas):
            costo = []
            for xi, objetivo in zip(estimulos_ext, respuestas):
               costo.append(self.actualizar_pesos(xi, objetivo))
            u_costo = sum(costo) / len(respuestas)
            self.costos.append(u_costo)

        return self

    """
    Se inicializa el vector pesos con valores aleatorios dentro de un rango
    determinado por los valores minimo y maximo. Ademas asocia el sesgo a la ultima coordenada
    del vector de pesos
    """
    def iniciarPesos(self, nro_entradas, sesgo, minimo, maximo):
        pesos_sin_sesgo = np.random.uniform(minimo, maximo, (1, nro_entradas))
        sesgo_transpuesto = np.atleast_2d(sesgo).T
        pesos = np.concatenate(( pesos_sin_sesgo, sesgo_transpuesto), axis=1)
        self.pesos = pesos[0]

    """
    Actualizacion estocastica de los pesos sinapticos
    """
    def actualizar_pesos(self, xi, objetivo):
        salida = self.activacion(self.neu_entrada(xi))
        error = objetivo - salida
        self.pesos += self.etha * xi.dot(error)
        costo = 0.5 * error**2
        return costo
    
    """
    Valor recibido por la nurona antes de pasar por la funcion de activacion
    """
    def neu_entrada(self, x):
        return np.dot(x, self.pesos)
    
    """
    Funcion de Activacion. En el caso del Adaline es la funcion lineal o identidad
    """
    def activacion(self, x):
        return x
    
    """
    Dado un estimulo, retorna el resultado de estimular la neurona
    componente_sesgo es un valor booleano para indicar si el estimulo recibido 
    contiene la coordenada asociada al sesgo
    """
    def evaluar( self, estimulo_a_evaluar, componente_sesgo=False ):
        if not componente_sesgo:
            estimulo = np.array([*estimulo_a_evaluar, 1])
        else:
            estimulo = estimulo_a_evaluar
        return self.activacion(self.neu_entrada(estimulo))

