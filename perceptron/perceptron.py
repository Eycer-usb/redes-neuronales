"""

Programa cliente de pruebas para la implementacion del 
Perceptron multiclases de Rosemblatt.

"""

from PerceptronDeRosemblatt import *
import numpy as np
import os.path

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
cada categoria. Luego se entrena al perceptron con tres tasas de aprendizaje dististas
y se imprime por la salida estandar el porcentaje de acierto del perceptron al clasificar 
un estimulo
"""
def clasificar( estimulo1, estimulo2, respuesta_deseada1, respuesta_deseada2):
    # Creando las variables para el perceptron
    estimulos = np.concatenate((estimulo1, estimulo2))
    respuestas_deseadas = np.concatenate((respuesta_deseada1, respuesta_deseada2))
    sesgo = np.array([1])
    tasa_aprendizaje = [0.001, 0.01, 0.1]
    minimo_peso = -0.05
    maximo_peso = 0.05
    max_epocas = 100

    # Por cada tasa de aprendizaje aplicamos el algoritmo
    for tasa in tasa_aprendizaje:
        #Creamos el Perceptron y lo entrenamos
        perceptron = PerceptronDeRosemblat( nro_neuronas = 1 )
        perceptron.entrenar( estimulos, respuestas_deseadas, sesgo, 
                            tasa, minimo_peso, maximo_peso, max_epocas)
        
        print(f"Porcentaje de Acierto con tasa de {tasa} : ", 
              perceptron.porcentaje_aciertos)
        
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

    ### Ciencias de la tierra y el espacio vs Ciencias medicas ###
    print("### Ciencias de la tierra y el espacio vs Ciencias medicas ###")
    clasificar(estimulos_tierra, estimulos_medicas, 
               respuesta_deseada_tierra, respuesta_deseada_medicas)

    ### Ciencias de la vida vs Agricultura ###
    print("### Ciencias de la vida vs Agricultura ###")
    clasificar(estimulos_vida, estimulos_agro,
               respuesta_deseada_vida, respuesta_deseada_agro)
    


if __name__ == '__main__':
    main()