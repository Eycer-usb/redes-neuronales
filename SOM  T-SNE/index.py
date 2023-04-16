"""
Se utiliza la implementacion de un Mapa Auto Organizativo SOM
y la implementacion de una TSE para visualizar todos los datos
provisto a lo largo del curso
"""

import utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn_som.som import SOM

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
    m = 10
    n = 10
    data = []
    for filename in files:
        data = data +  list(utils.get_data(filename))
    data = np.array(data)
    epochs_ = [ 5, 10, 100, 1000]
    fig,ax = plt.subplots(nrows=1, ncols=4, figsize=(15,4))
    for i, e in enumerate(epochs_):
        som = SOM(m=m, n=n, dim=512, lr=0.1, random_state=12346)
        som.fit(data, epochs=e)
        predictions = som.predict(data)
        G = np.zeros((m,n,4))
        for p in predictions:
            x = p%n
            y = p//n
            G[x][y] = [20,0,255, G[x][y][3]+0.5]
        print(G[:,:,3])
        ax[i].imshow(G.astype(int))
        ax[i].set_title(f"Epocas {e}")
     
        
    plt.show()
    # print(list(predictions))
    
    
    
    
        

if __name__ == '__main__':
    main()