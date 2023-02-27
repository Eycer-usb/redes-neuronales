"""
Implementacion del Perceptron multiclases de Rosemblat
para la clasificacion de datos linealmente separables
"""

import numpy as np

# Clase Principal
class PerceptronDeRosemblat:

    # Funcion constructora de la clase
    def __init__(self, nro_neuronas):
        self.nro_neuronas = nro_neuronas
        self.epocas = 1
        self.pesos = None
        self.porcentaje_aciertos = None
        self.fun_activacion = np.vectorize(self.sgn)

    # Metodo de Entrenamiento de las neuronas
    def entrenar(self, estimulos, respuestas_deseadas, 
                 sesgo, tasa_aprendizaje, minimo_peso=0, 
                 maximo_peso=1, max_epocas=None):
        
        # Inicializacion de las variables del algoritmo
        (nro_estimulos, nro_entradas)  = estimulos.shape
        self.pesos = self.iniciarPesos(nro_entradas, sesgo, minimo_peso, maximo_peso)
        estimulos_ext = np.concatenate((estimulos, np.atleast_2d(np.ones(nro_estimulos)).T), axis=1)
        i = 0
        hubo_ajuste = False
        aciertos = 0

        # Se ejecuta el entrenamiento hasta que se clasifiquen correctamente todos
        # los estimulos o hasta alcanzar el maximo de epocas especificado
        while i < nro_estimulos and (max_epocas == None or self.epocas < max_epocas):
            
            # Calculando el valor de entrada en el nucleo de la neurona
            valor_recibido_en_nucleo = np.matmul(self.pesos,(np.atleast_2d(estimulos_ext[i]).T))
            
            # Aplicando la funcion de activacion 'Signo' vectorizada sobre el valor recibido en el
            # nucleo de la neurona
            respuesta_obtenida = np.atleast_2d(self.fun_activacion(valor_recibido_en_nucleo)).T[0]

            # Si la clasificacion es incorrecta se ajustan los pesos
            if( (respuesta_obtenida != respuestas_deseadas[i]).any() ):
                error = respuestas_deseadas[i] - respuesta_obtenida
                self.pesos += tasa_aprendizaje*error*estimulos_ext[i]
                hubo_ajuste = True
            # Si no no se realiza ninguna modificacion
            else:
                aciertos+=1
            i+=1

            # Se lleva el conteo de los estimulos acertados en cada epoca
            if (i == nro_estimulos and 
                (max_epocas == None or (self.epocas + 1 < max_epocas)) and
                hubo_ajuste):
                aciertos = 0
            
            # Se lleva la cuenta de las epocas de entrenamiento
            if( i == nro_estimulos and hubo_ajuste):
                i = 0
                hubo_ajuste = False
                self.epocas += 1

        # Calculo y almacenamiento del porcentaje de aciertos
        self.porcentaje_aciertos = aciertos*100/nro_estimulos

        return self.pesos

    # Se inicializa el vector (o matriz) de pesos para la primera iteracion del 
    # algoritmo
    def iniciarPesos(self, nro_entradas, sesgo, minimo, maximo):
        pesos_sin_sesgo = np.random.uniform(minimo, maximo, (self.nro_neuronas, nro_entradas))
        sesgo_transpuesto = np.atleast_2d(sesgo).T
        pesos = np.concatenate(( pesos_sin_sesgo, sesgo_transpuesto), axis=1)
        np.random.uniform()
        return pesos
    
    # Funcion signo (sin vectorizar)
    def sgn( self, xw ):
        if xw>=0:
            return 1
        else:
            return -1