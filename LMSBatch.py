"""
Implementacion del Perceptron multiclases de Rosemblat
para la clasificacion de datos linealmente separables
"""

import numpy as np

# Clase Principal
class LMSBatch:

    # Funcion constructora de la clase
    def __init__(self, nro_neuronas):
        self.nro_neuronas = nro_neuronas
        self.epocas = 1
        self.pesos = None
        self.porcentaje_aciertos = None
        self.error = float('inf')

    # Metodo de Entrenamiento de las neuronas
    def entrenar(self, estimulos, respuestas_deseadas, 
                 sesgo, tasa_aprendizaje_inicial, minimo_peso=0, 
                 maximo_peso=1, max_epocas=100):
        
        # Asercion de maximo de epocas correcta
        assert(max_epocas > 0)
        
        # Inicializacion de las variables del algoritmo
        (nro_estimulos, nro_entradas)  = estimulos.shape
        self.pesos = self.iniciarPesos(nro_entradas, sesgo, minimo_peso, maximo_peso)
        estimulos_ext = np.concatenate((estimulos, np.atleast_2d(np.ones(nro_estimulos)).T), axis=1)
        for q in range(max_epocas):
            # print(f"Epoca {q}")
            error = self.suma_errores(estimulos_ext, respuestas_deseadas)
            # print("DIFERENCIA: ", error)
            for i in range(nro_estimulos):
                self.pesos += tasa_aprendizaje_inicial*error*estimulos_ext[i]
                out = self.obtenido(estimulos_ext)
                # print("OBTENIDO:", out)
                # print("ESPERADO:", respuestas_deseadas)
                # print("ERROR: ", self.error_cuadratico_medio( respuestas_deseadas, out ))
            self.epocas+=1


        self.error = self.error_cuadratico_medio( respuestas_deseadas, self.obtenido(estimulos_ext) )
        self.porcentaje_aciertos = self.acertados(respuestas_deseadas, self.obtenido(estimulos_ext), 1 )*100/nro_estimulos
        return self.pesos

    def obtenido(self, X):
        ( filas, _) = X.shape
        ans = np.zeros( (filas, 1 ))
        for i in range(filas):
            ans[i] = np.dot( X[i], self.pesos )
        return ans.T[0]
    
    def activation(self, X):
        return X
    # Se inicializa el vector (o matriz) de pesos para la primera iteracion del 
    # algoritmo
    def iniciarPesos(self, nro_entradas, sesgo, minimo, maximo):
        pesos_sin_sesgo = np.random.uniform(minimo, maximo, (self.nro_neuronas, nro_entradas))
        sesgo_transpuesto = np.atleast_2d(sesgo).T
        pesos = np.concatenate(( pesos_sin_sesgo, sesgo_transpuesto), axis=1)
        return pesos[0]
    
    # Funcion signo (sin vectorizar)
    def indentity( self, xw ):
        return xw
    
    def suma_errores( self, estimulos, deseado ):
        obtenido = self.obtenido(estimulos)
        return (deseado - obtenido ).sum()
    
    def error_cuadratico_medio( self, deseado, obtenido):
        return abs(deseado - obtenido).sum()
    
    def acertados( self, deseado, obtenido, delta ):
        acertados = 0
        (l,) = deseado.shape
        for i in range(l):
            if abs( deseado[i]- obtenido[i] ) < delta : acertados+=1
        return acertados