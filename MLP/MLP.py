from PerceptronLayer import *

class MLP:
    """
    Arguments
    learning_rate: Learning rate of the network
    activation_function: F(vi) Activation function, it will be vectorizate
    first_derivative: F'(vi) first derivative of activation function
    input_dimentions: Dimentions of each stimul vector
    neuron_dimentions: Tuple of structure for the network. For example:
                        (4,3,2) for a 4x3x2 network
    """
    def __init__(self, learning_rate, activation_function, first_derivative, 
                 input_dimentions, neuron_dimentions ):
        self._etha = learning_rate
        self._fn = activation_function
        self._dfn = first_derivative
        self._input_d = input_dimentions
        self._neuron_d = neuron_dimentions

        # Initializating Network
        self._network = []
        layer_number = 0
        layer_input_number = input_dimentions
        for dimention in neuron_dimentions:
            layer = PerceptronLayer(layer_input_number, dimention, 
                                    activation_function, first_derivative, 1,
                                    learning_rate, layer_number)
            self._network.append(layer)
            layer_number += 1
            layer_input_number = dimention

    
    def _train_input(self, input_vector_no_bias, expected_answer):
        # Activating Network
        input = input_vector_no_bias
        for layer in self._network:
            input = layer.activate(input)
        # At the last activation the input variable will contain 
        # the output of the entire network

        # Backpropagation
        N = len(self._network)
        for i in range( N-1, -1, -1):
            if i + 1 != N :
                layer[i].train(input_vector_no_bias, None, layer[i+1])
            else:
                layer.train(input_vector_no_bias, expected_answer)

    """
    Training Function
    input_array: A python 2 dimentional List with the stimuls
    expected_answers: A 2D List with expected answers
    max_epoch: Max Epoch to train network 
    """
    def train(self, inputs_array, expected_answers, max_epoch):
        for i in range(max_epoch):
            N = len(inputs_array)
            for i in range(N):
                input_vector_no_bias = np.array(inputs_array[i])
                expected_answer = np.array(expected_answers[i])
                self._train_input(input_vector_no_bias, expected_answer )
        
            



