import numpy as np
class LMSStochastic:
    def __init__(self):
        self.error = 0
        self.costos = []
        self.pesos = []

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

    def iniciarPesos(self, nro_entradas, sesgo, minimo, maximo):
        pesos_sin_sesgo = np.random.uniform(minimo, maximo, (1, nro_entradas))
        sesgo_transpuesto = np.atleast_2d(sesgo).T
        pesos = np.concatenate(( pesos_sin_sesgo, sesgo_transpuesto), axis=1)
        self.pesos = pesos[0]

    def actualizar_pesos(self, xi, objetivo):
        salida = self.activacion(self.neu_entrada(xi))
        error = objetivo - salida
        self.pesos += self.etha * xi.dot(error)
        costo = 0.5 * error**2
        return costo
    
    def neu_entrada(self, x):
        return np.dot(x, self.pesos)
    
    def activacion(self, x):
        return x
    
    def evaluar( self, estimulo_a_evaluar, componente_sesgo=True ):
        if not componente_sesgo:
            estimulo = np.array([*estimulo_a_evaluar, 1])
        else:
            estimulo = estimulo_a_evaluar
        return self.activacion(self.neu_entrada(estimulo))

