"""
Se utiliza la implementacion de un Mapa Auto Organizativo SOM
y la implementacion de una TSE para visualizar todos los datos
provisto a lo largo del curso
"""

import utils
import numpy as np
import matplotlib.pyplot as plt
from sklearn_som.som import SOM
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns

"""
Visualizaciones mediante t_sne
"""
def t_sne_(data):
    n_components = 2
    tsne = TSNE(n_components)
    tsne_result = tsne.fit_transform(data)
    y = ['Agri']*2078 + ['Astro']*475 + ['Chem']*572 + ['Earth-Space']*389 + \
        ['LifeSci']*1778 + ['Math']*185 + ['MedSci']*330 + ['Physic']*125 + ['TechSci']*533
    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': y})
    fig, ax = plt.subplots(1)
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax,s=120)
    lim = (tsne_result.min()-5, tsne_result.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.show()
"""
Visualizaciones mediante som
"""
def som_(data, m, n):
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

"""
En el programa principal se ejecutan las pruebas y visualizaciones de los datos
a travez de dos dispositivos diferentes:
- SOM: Self Organizing Maps
- TSE: T-distributed stochastic neighbour embedding 
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
    data = []
    for filename in files:
        data = data +  list(utils.get_data(filename))
    data = np.array(data)
    m = 10
    n = 10
    som_(data, m, n )
    t_sne_(data)
    
    
    
    
    
        

if __name__ == '__main__':
    main()