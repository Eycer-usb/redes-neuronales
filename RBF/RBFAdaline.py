"""
Red RBF con eleccion de centros por medio del metodo de las k-medias
"""
import numpy as np
from sklearn.cluster import KMeans
from UniAprox import *
class RBFAdaline:
    """
    Se crean los centroides y se resuelve el sistema de ecuaciones asociado
    """
    def __init__(self, num_neuron, data, function, factor_regulation = 0, learning_rate=0.01, epoch=1000, bias = 1 ) -> None:
        self.num_neuron = num_neuron
        self.factor = factor_regulation
        self.function = function
        self.data = data
        self.kmeans = KMeans( init="k-means++", n_clusters=num_neuron, n_init=10, max_iter=300)
        self.kmeans.fit( data )
        self.centers = self.kmeans.cluster_centers_
        self.center_x = np.array([[center[0] for center in self.centers ]]).T
        self.center_y = np.array([[center[1] for center in self.centers ]]).T
        self.epoch = epoch
        self.bias = 1
        self.learning_rate = learning_rate
        self.weight = np.random.uniform(-0.05, 0.05, (num_neuron))
        self.interpolator = self.train()
        self.inertia = self.kmeans.inertia_
    
    
    def train(self):
        n = self.epoch
        lr = self.learning_rate
        for _ in range(n):
            for x, d in zip(self.data[:,0], self.data[:,1]):
                phi = np.array([ self.function(self.distance(x,c)) for c in self.centers ])
                y = np.dot(self.weight, phi)
                error = (d - y)
                self.weight += self.learning_rate*phi*error
                self.bias += self.learning_rate * error

    
    def interpolate(self, x):
        phi = np.array([ self.function(self.distance(x,c)) for c in self.centers ])
        y = np.dot(self.weight, phi)
        return y
    
    def distance(self, np_vector1, np_vector2) -> float:
        return np.linalg.norm(np_vector1 - np_vector2)
        
"""
Funcion Gaussiana
"""
dev = -1
def gaussian(x):
    return np.exp( -1/(2*dev**2) * x )

def main():
    import client
    global dev
    ( inputs, expected, dev )   = client.get_training_set( "../datos/Spectra100 - Spectra100.csv" )
    data = np.concatenate((inputs, expected), axis=1)
    rbf = RBFAdaline(50, data, gaussian)
    
    client.plot(inputs, expected, rbf, f"RBF Adaline con {50} neuronas lambda = {0}", "", "", True)
    # get_metric(rbf)
if __name__ == '__main__':
    main()
    

        
