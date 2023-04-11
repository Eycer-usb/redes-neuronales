"""
Red RBF con eleccion de centros por medio del metodo de las k-medias
Se reciben como parametros:
- 


"""
import numpy as np
from sklearn.cluster import KMeans
from UniAprox import *
class RBF:
    def __init__(self, num_neuron, data, function, factor_regulation = 0) -> None:
        self.num_neuron = num_neuron
        self.factor = factor_regulation
        self.function = function
        self.data = data
        self.kmeans = KMeans( init="k-means++", n_clusters=num_neuron, n_init=10, max_iter=300)
        self.kmeans.fit( data )
        self.centers = self.kmeans.cluster_centers_
        center_x = np.array([[center[0] for center in self.centers ]]).T
        center_y = np.array([[center[1] for center in self.centers ]]).T
        print('center_x', center_x)
        print('center_y', center_y)
        self.interpolator = UniAprox(center_x, center_y, function, factor_regulation )
    
    
    def get_weights(self):
        return self.adaline.pesos
    
    def interpolate(self, x):
        return self.interpolator.interpolate( x )
    
    

        
