"""

Programa cliente de pruebas para la implementacion del 
Adaline LMS.

"""
import os.path
from LMSBatch import *
from LMSStochastic import *
import numpy as np
import matplotlib.pyplot as plt
from random import randint

"""
Esta funcion recibe la ruta a un archivo .csv
y retorna una matriz de floats con los correspondientes
valores almacenados en el archivo.
"""
def obtener_estimulos( nombre_archivo ):
    with open(f'{os.path.dirname(__file__)}/../{nombre_archivo}', "r", encoding='utf-8') as f:
        lineas = f.readlines()
        return np.array(list(map( lambda linea: 
                     list( map( lambda dato: float(dato), linea.split(',')) ),
                    lineas)))

"""
Al clasificar se reciben los estimulos y respuestas deseadas de 
cada categoria. Luego se entrena al adaline con 4 tasas de aprendizaje dististas
y se imprime por la salida estandar el porcentaje de acierto del adaline al clasificar 
un estimulo
"""
       
def clasificar( estimulos, respuestas_deseadas, tasa_aprendizaje ):
    max_epocas = 100
    pesos = []
    errores = []
    # Por cada tasa de aprendizaje aplicamos el algoritmo
    for tasa in tasa_aprendizaje:
        #Creamos el lms y lo entrenamos
        lms = LMSStochastic()
        lms.entrenar(np.array(estimulos), respuestas_deseadas, tasa, max_epocas)
        errores.append(lms.costos[-1])
        print(f"Error cuadratico medio con tasa: {tasa}\n", lms.costos[-1])
        pesos.append(list(lms.pesos))
    return pesos, errores
"""
Se recibe un valor para la variable x y se retorna por cada una un vector
con los valores [ x, x^2, x^3, x^4 ]
"""
def construir_estimulo_polinomico( estimulos, grado ):
    ans = []
    for estimulo in estimulos:
        p = []
        for i in range(grado, 0, -1):
            p.append(estimulo**grado)
        ans.append(p)
    return np.array(ans)

"""
Funcion principal y punto de arranque del cliente
"""
def main():

    # Se leen los datos de los archivos correpondientes
    # y se arman las estructuras de los estimulos y las 
    # respuestas deseadas para cada clase

    # Ciencias de la Tierra
    estimulos_tierra = obtener_estimulos('datos/EarthSpace - EarthSpace.csv')
    (nro_estimulos, _ ) = estimulos_tierra.shape
    respuesta_deseada_tierra = np.zeros(nro_estimulos) + 1

    # Ciencias Medicas
    estimulos_medicas = obtener_estimulos('datos/MedSci - MedSci.csv')
    (nro_estimulos, _ ) = estimulos_medicas.shape
    respuesta_deseada_medicas = np.zeros(nro_estimulos) - 1

    # Ciencias de la vida
    estimulos_vida = obtener_estimulos('datos/LifeSci - LifeSci.csv')
    (nro_estimulos, _ ) = estimulos_vida.shape
    respuesta_deseada_vida = np.zeros(nro_estimulos) + 1

    # Agricultura
    estimulos_agro = obtener_estimulos('datos/Agri - Agri.csv')
    (nro_estimulos, _ ) = estimulos_agro.shape
    respuesta_deseada_agro = np.zeros(nro_estimulos) - 1

    # Interpolador
    estimulos_interpolador = obtener_estimulos('datos/datosT3 - datosT3.csv')
    (nro_estimulos, _) = estimulos_interpolador.shape
    respuesta_deseada_interpolador = estimulos_interpolador[:,1]
    estimulos_interpolador =np.atleast_2d(estimulos_interpolador[:,0]).T

    ### Ciencias de la tierra y el espacio vs Ciencias medicas ###
    print("### Ciencias de la tierra y el espacio vs Ciencias medicas ###")
    estimulos = np.concatenate((estimulos_tierra, estimulos_medicas))
    respuestas_deseadas = np.concatenate((respuesta_deseada_tierra, respuesta_deseada_medicas))
    clasificar(estimulos, respuestas_deseadas, tasa_aprendizaje = [ 0.001, 0.01, 0.1, 0.25 ])

    # ### Ciencias de la vida vs Agricultura ###
    print("\n### Ciencias de la vida vs Agricultura ###")
    estimulos = np.concatenate((estimulos_vida, estimulos_agro))
    respuestas_deseadas = np.concatenate((respuesta_deseada_vida, respuesta_deseada_agro))
    clasificar(estimulos, respuestas_deseadas, tasa_aprendizaje = [ 0.001, 0.01, 0.1, 0.25 ])

    ### Interpolador ###

    # Mediante Funcion Lineal
    print("\n### Interpolador Lineal ###")
    pesos = clasificar(estimulos_interpolador, respuesta_deseada_interpolador, tasa_aprendizaje = [ 0.001, 0.01, 0.1, 0.25 ] )


    ### Graficacion ###
    estimulos_interpolador = obtener_estimulos('datos/datosT3 - datosT3.csv')
    plt.plot(estimulos_interpolador[:,0], estimulos_interpolador[:, 1], 'ro')
    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')
    plt.show()

    x = np.linspace(-2,2,100)
    y = pesos[-1][0]*x + pesos[-1][1]
    plt.plot(x, y, 'g')
    plt.plot(estimulos_interpolador[:,0], estimulos_interpolador[:, 1], 'ro')
    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')
    plt.show()
    
    
    # Funcion polinomica de grado 3
    print("\n### Interpolador Polinomico grado 3 ###")
    estimulos_interpolador = obtener_estimulos('datos/datosT3 - datosT3.csv')
    p = construir_estimulo_polinomico(estimulos_interpolador[:,0], grado=3 )
    respuesta_deseada_interpolador = estimulos_interpolador[:,1]
    tasas_aprendizaje = [ 0.01, 0.001, 0.0001, 0.00001 ]
    pesos, errores = clasificar(p, respuesta_deseada_interpolador, tasa_aprendizaje = tasas_aprendizaje)
    colores = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    # Graficacion
    plt.xlabel('Eje X')
    plt.ylabel('Eje Y')
    plt.title( f'Polinomios Grado 3' )
    plt.plot(estimulos_interpolador[:,0], estimulos_interpolador[:, 1], 'ro')
    estimulos_interpolador = obtener_estimulos('datos/datosT3 - datosT3.csv')
    x = np.linspace(-3,3,100)
    for peso, tasa, error in zip(pesos, tasas_aprendizaje, errores):
        i = randint(0, len(colores)-1)
        y = peso[0]*(x**3) + peso[1]*(x**2) + peso[2]*(x) + peso[3]
        print(peso)
        plt.plot(x, y, colores.pop(i), label=f'Etha: {tasa}, Error: {round(error,3)}')
    plt.legend(loc="upper left")
    plt.show()

    

if __name__ == '__main__':
    main()