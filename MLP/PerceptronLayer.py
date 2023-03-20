import numpy as np
class PerceptronLayer:
    """
    Arguments:
    input number: Dimension of input vector
    neuron_number: Number of neuron to create in this layer
    activation_function: function to apply to activate the neuron in this layer
    first_derivative: First derivative of the activation function
    bias: A vector with the bias for each neuron, or a number with the default value
    layer_name: Becouse everybody deserves a name

    For a better use, the weith storange is transposed, but it can be get using the get_untransposed_weight method
    """
    def __init__(self, input_number, neuron_number,
                activation_function, first_derivative_function,
                bias, learning_rate, layer_name="", alpha=0):
        self._input_n = input_number
        self._neuron_n = neuron_number
        self._name = layer_name
        self._etha = learning_rate
        self._alpha = alpha
        self._momentum = 0
        _bias = bias
        if( type(bias) == int ):
            _bias = np.zeros((neuron_number,1)) + bias
        initial_weight = np.random.uniform(-0.005,0.005,(neuron_number, input_number))
        self._weight = np.concatenate( (initial_weight, _bias ), axis=1 )
        self._fn = np.vectorize(activation_function)
        self._dfn = np.vectorize(first_derivative_function)

    def _activate(self, input_vector):
        # print(f"{self._weight} x {input_vector}={np.matmul(self._weight, input_vector)}" )
        try:
            salida = self._fn( np.matmul(self._weight, input_vector) )
            return salida
        except:
            raise Exception(f"Error activating neurons layer {self._name}")
    
    def activate(self, input_vector_no_bias):
        # print(f"{self._weight} x {input_vector}={np.matmul(self._weight, input_vector)}" )
        input_vector = np.concatenate( (input_vector_no_bias, np.array([1])), axis=None )
        try:
            salida = self._fn( np.matmul(self._weight, input_vector) )
            return salida
        except:
            raise Exception(f"Error activating neurons layer {self._name}")

    def get_untransposed_weight(self):
        return self._weight.T
    
    def get_local_gradient_sum(self, next_layer):
        gradient_sum = []
        # print(next_layer._weight, next_layer.local_gradient)
        # print(next_layer._input_n, next_layer._neuron_n)
        for j in range(next_layer._input_n):
            sum_j = 0
            for alpha in range(next_layer._neuron_n):
                sum_j += next_layer._weight[alpha][j] * next_layer.local_gradient[alpha]
            gradient_sum.append(sum_j)
        return np.array(gradient_sum)


    def train(self, input_vector_no_bias, expected_answer, next_layer=None):
        input_vector = np.concatenate( (input_vector_no_bias, np.array([1])), axis=None )
        if not next_layer:
            answer = self._activate(input_vector)
            e = expected_answer - answer
            dfn_v = self._dfn( np.matmul(self._weight, input_vector) )
            local_gradient = np.array([np.multiply(e, dfn_v)])
            # print(np.matmul( local_gradient.T , np.array([input_vector]) ))
            delta_weight = self._etha * np.matmul( local_gradient.T , np.array([input_vector]) )
            self._weight +=  delta_weight + self._momentum
            self._momentum = self._alpha*delta_weight
            self.local_gradient = local_gradient[0]
        else:
            dfn_v = self._dfn( np.matmul(self._weight, input_vector) )
            grad_sum = self.get_local_gradient_sum(next_layer)
            local_gradient = np.array([np.multiply(  grad_sum, dfn_v )])
            delta_weight = self._etha * np.matmul( local_gradient.T , np.array([input_vector]) )
            self._weight += delta_weight + self._momentum
            self._momentum = self._alpha*delta_weight
            self.local_gradient = local_gradient[0]
        