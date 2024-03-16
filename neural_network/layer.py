import pandas
import numpy
from neural_network.activation_functions import ActivationFunction


class Layer:
    """
    A layer consists of:
        * the number of nodes in the layer
        * an array of incoming inputs
        * the number of outputs
        * the activation function
    The Layer will then:
        * have a list of weights for the incoming inputs
        * calculate the node value based on the incoming inputs, weights, and bias
        * pass the node value through the activation equation
        * create an output array of inputs for the next layer
    """

    def __init__(
        self,
        name: str,
        nodes: int,
        number_of_inputs: int,
        number_of_outputs: int,
        activation_function: ActivationFunction,
        # activation_function_coefficient: float = 0,
    ) -> None:
        # generate
        self.name = name
        self.bias = numpy.random.default_rng().normal(loc=0, scale=0.25, size=nodes)
        self.weighted_inputs = numpy.random.default_rng().normal(
            loc=0, scale=0.25, size=(nodes, number_of_inputs)
        )
        self.weighted_velocities = numpy.ones((nodes, number_of_inputs))
        self.number_of_outputs = number_of_outputs
        self.activation_function = activation_function

    def update_weights(self, node_index: int, weight_adjustments):
        self.weighted_inputs[node_index] += weight_adjustments

    def update_bias(self, bias_adjustments):
        self.bias += bias_adjustments

    def get_outputs(self, activation_values: pandas.DataFrame):
        """
        Take the weights times inputs with the bias added to get the outputs (i.e. inputs for the next node)
            weights (n,j)           *    inputs (j,1)   +       bias       =   outputs (n,1)
        Γ                       Ꞁ        Γ         Ꞁ         Γ         Ꞁ       Γ         Ꞁ
          w_1_1 w_2_1 ... w_j_1              i_1                b_1                o_1
          w_2_2 w_2_2 ... w_j_2     *        i_2        +       b_2        =       o_2
                  ...                        ...                ...                ...
          w_n_1 w_n_2 ... w_j_n              i_j                b_j                o_n
        L                       ⅃       L          ⅃        L          ⅃       L          ⅃
        """
        # node_values = numpy.matmul(self.weighted_inputs, inputs) + self.bias
        activation_function_vector = numpy.vectorize(self.activation_function.equation)
        return activation_function_vector(activation_values)

    def get_derivative(self, activation_values: pandas.DataFrame):
        activation_function_derivative_vector = numpy.vectorize(
            self.activation_function.derivative
        )
        return activation_function_derivative_vector(activation_values)
