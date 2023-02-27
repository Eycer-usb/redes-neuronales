"""

Programa cliente de pruebas para la implementacion del 
Adaline LMS.

"""

from LMSBatch import *
from LMSStochastic import *
import numpy as np
import matplotlib.pyplot as plt

"""
Esta funcion recibe la ruta a un archivo .csv
y retorna una matriz de floats con los correspondientes
valores almacenados en el archivo.
"""
def obtener_estimulos( nombre_archivo ):
    with open(f'{nombre_archivo}', "r", encoding='utf-8') as f:
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
       
def clasificar( estimulos, respuestas_deseadas ):
    tasa_aprendizaje = [ 0.001, 0.01, 0.1, 0.25 ]
    max_epocas = 100
    pesos = []
    # Por cada tasa de aprendizaje aplicamos el algoritmo
    for tasa in tasa_aprendizaje:
        #Creamos el lms y lo entrenamos
        lms = LMSStochastic()
        lms.entrenar(np.array(estimulos), respuestas_deseadas, tasa, max_epocas)
        print(f"Error cuadratico medio con tasa: {tasa}\n", lms.costos[-1])
        pesos.append(list(lms.pesos))
    return pesos
        
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
    clasificar(estimulos, respuestas_deseadas)

    # ### Ciencias de la vida vs Agricultura ###
    print("\n### Ciencias de la vida vs Agricultura ###")
    estimulos = np.concatenate((estimulos_vida, estimulos_agro))
    respuestas_deseadas = np.concatenate((respuesta_deseada_vida, respuesta_deseada_agro))
    clasificar(estimulos, respuestas_deseadas)

    ### Interpolador ###
    print("\n### Interpolador ###")
    pesos = clasificar(estimulos_interpolador, respuesta_deseada_interpolador)
    

    ### GRAFICACION ###
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
    
    
    

if __name__ == '__main__':
    main()