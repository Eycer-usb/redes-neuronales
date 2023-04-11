"""
Red RBF con eleccion de centros por medio del metodo de las k-medias
Se reciben como parametros:
- 


"""
import numpy as np
from sklearn.cluster import KMeans
class RBF:
    def __init__(self, num_neuron, data, function, factor_regulation = 0) -> None:
        self.num_neuron = num_neuron
        self.factor = factor_regulation
        self.function = function
        self.kmeans = KMeans( init="k-means++", n_clusters=num_neuron, n_init=10, max_iter=300)
        self.kmeans.fit( data )
        self.centers = self.kmeans.cluster_centers_

    def distance(self, np_vector1, np_vector2) -> float:
        return np.linalg.norm(np_vector1 - np_vector2)
    
    

        
