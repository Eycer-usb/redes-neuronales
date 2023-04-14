"""
Clasificacion de los datos recibidos mediante SVM

Este programa realiza un estudio de los datos de prueba recibidos a lo largo
del curso para generar  Maquinas de Vectores de Soporte
o SVM por sus siglas en ingles capaces de resolver los dos problemas de clasificacion
de textos variando el kernel (se prueban dos kernels) y ajustando los parametros del modelo
lo mejor posible

Se utiliza la libreria Scikit-learn de python especificamente la clase SVM.

"""
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import statistics as stat
import numpy as np
import csv

"""
Se lee un archivo de datos y se retorna el conjunto de estimulos

Argumentos:
- filename: Nombre del archivo

Respuesta:
- Arreglo bidimensional de estimulos
"""
def get_data(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        csvreader = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        lines = []
        for row in csvreader:
            lines.append(row)
        lines = np.array(lines)
        inputs = lines[:, [0]]
        return np.array(inputs)

"""
Se Clasifican los estimulos creando una maquina SVM
Argumentos:
- Stimulos: Matriz bidimensional de estimulos de tipo np.array
- Expected: Vector fila de valores esperados de tipo np.array

Funcionalidad:
- Se separan de los estimulos un conjunto de entrenamiento que consta de un 
80% de los datos y un conjunto de validacion con el el restante 20% para cuantificar
la capacidad de prediccion y estimacion de la red
- La metrica utilizada es la cantidad de datos bien clasificados del conjunto de validacion
- Se iteran sobre diferentes parametros y kernels en busca de aquellos con mejor metrica

Retorno:
- El error cuadratico medio cometido en la prediccion del conjunto de validacion
"""
def svm_classificate_and_compare( stimuls, expected ):
    n = stimuls.shape[0]
    sample = np.random.choice(n, size=n*80//100, replace=False)
    train_set = stimuls[sample, :]
    train_set_exp = expected[sample]
    validation_set = np.delete(stimuls, sample, 0)
    validation_set_exp = np.delete(expected, sample)
    param_grid = { 
        'kernel':['linear', 'rbf', 'sigmoid', 'poly'],
        'coef0':[ 1 ],
        'C': [ 0.1, 1, 10, 20],
        'gamma': [0.001, 0.01, 0.1, 1, 5],
        'degree': [ 10, 15, 20, 25, 50 ]
    }


    cv = KFold(shuffle=True)
    grid = GridSearchCV(svm.SVC(), param_grid=param_grid, cv=cv, verbose=3)
    grid.fit(train_set, train_set_exp)
    predicted = grid.predict(validation_set)
    print(grid.best_params_)
    print(grid.best_score_)

    return metrics.accuracy_score(validation_set_exp, predicted)




"""
Funcion de inicio del programa de pruebas
"""
def main():

    """
    Se generan los pares de archivos de texto a comparar.
    Se compararan EarthSpace vs MedSci  y Agri vs LifeSci
    """
    classifications = [
        [ '../datos/EarthSpace - EarthSpace.csv', '../datos/MedSci - MedSci.csv' ],
        [ '../datos/Agri - Agri.csv', '../datos/LifeSci - LifeSci.csv' ],
    ]

    print("\n=== Porcentaje de Bien clasificados en la prediccion del conjunto de validacion ===")
    for vector in classifications:
        """
        Se obtienen los estimulos de cada archivo y se generan las respuestas esperadas
        para luego unir en una sola estructura los estimulos y en otra las respuestas
        esperadas
        """
        stimuls_1 = get_data( vector[0] )
        stimuls_2 = get_data( vector[1] )
        expected_1 = np.zeros((stimuls_1.shape[0])) - 1
        expected_2 = np.ones((stimuls_2.shape[0]))

        stimuls = np.concatenate((stimuls_1, stimuls_2))
        expected = np.concatenate((expected_1, expected_2))
        
        avg = svm_classificate_and_compare( stimuls, expected )


if __name__ == "__main__":
    main()