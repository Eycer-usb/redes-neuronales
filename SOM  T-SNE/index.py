"""
Se utiliza la implementacion de un Mapa Auto Organizativo SOM
y la implementacion de una TSE para visualizar todos los datos
provisto a lo largo del curso
"""

import utils
import numpy as np
import matplotlib.pyplot as plt
from som import *
import random as rand

"""
Visualizaciones mediante t_sne
"""
def t_sne_():
    pass

"""
Visualizaciones mediante som
"""
def som_(som, data):
    pass

"""
En el programa principal se ejecutan las pruebas y visualizaciones de los datos
a travez de dos dispositivos diferentes:
- SOM: Self Organizing Maps
- TSE: T-distributed stochastic neighbor embedding 
"""
def main():
    files = [
        '../datos/Agri.csv',
        '../datos/AstroAstro.csv',
        '../datos/Chem.csv',
        '../datos/EarthSpace.csv',
        '../datos/LifeSci.csv',
        '../datos/Math.csv',
        '../datos/MedSci.csv',
        '../datos/Physic.csv',
        '../datos/TechSci.csv',
    ]
    m = 20
    n = 20
    data = []
    for filename in files:
        data = data +  list(utils.get_data(filename))
    train_data = np.array(data)
    n_x = train_data.shape[0]
    dim = train_data.shape[1]
    
    rand = np.random.RandomState(0)
    
    # Initialize the SOM randomly
    SOM = rand.randint(0, 255, (m, n, dim)).astype(float)

    # Display both the training matrix and the SOM grid
    fig, ax = plt.subplots(
    nrows=1, ncols=4, figsize=(15, 3.5), 
    subplot_kw=dict(xticks=[], yticks=[]))
    total_epochs = 0
    for epochs, i in zip([1, 4, 5, 10], range(0,4)):
        total_epochs += epochs
        SOM = train_SOM(SOM, train_data, epochs=epochs)
        RGB = vector_to_rgb(SOM.astype(int))
        ax[i].imshow()
        ax[i].title.set_text('Epochs = ' + str(total_epochs))
        
    plt.show()
    
def main2():
    # Dimensions of the SOM grid
    m = 5
    n = 4
    # Number of training examples
    n_x = 3000
    rand = np.random.RandomState(0)
    # Initialize the training data
    train_data = rand.randint(0, 255, (n_x, 3))
    
    # Initialize the SOM randomly
    SOM = rand.randint(0, 255, (m, n, 3)).astype(float)

    # Display both the training matrix and the SOM grid
    fig, ax = plt.subplots(
        nrows=1, ncols=2, figsize=(12, 3.5), 
        subplot_kw=dict(xticks=[], yticks=[]))
    ax[0].imshow(train_data.reshape(50, 60, 3))
    ax[0].title.set_text('Training Data')
    ax[1].imshow(SOM.astype(int))
    ax[1].title.set_text('Randomly Initialized SOM Grid')
    fig, ax = plt.subplots(
    nrows=1, ncols=4, figsize=(15, 3.5), 
    subplot_kw=dict(xticks=[], yticks=[]))
    total_epochs = 0
    for epochs, i in zip([1, 4, 5, 10], range(0,4)):
        total_epochs += epochs
        SOM = train_SOM(SOM, train_data, epochs=epochs)
        ax[i].imshow(SOM.astype(int))
        ax[i].title.set_text('Epochs = ' + str(total_epochs))
        
    plt.show()
    
    
    
        

if __name__ == '__main__':
    main()