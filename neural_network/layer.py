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
    A = activation function
    α = activation energy
    z = activation value
    w = weight
    """

    def __init__(
        self,
        name: str,
        number_of_inputs: int,
        number_of_outputs: int,
        activation_function: ActivationFunction,
        # activation_function_coefficient: float = 0,
    ) -> None:
        # generate
        self.name = name
        # The number of biases is equal to number of outputs
        self.bias = numpy.random.default_rng().normal(
            loc=0, scale=0.25, size=number_of_outputs
        )
        self.bias_velocities = numpy.zeros((number_of_outputs))
        # The number of weights is equal to the number of inputs times the number of outputs
        # Seeding the weighted inputs by grabbing values from a normal distribution with mean of 0
        # size is rows, columns
        self.weights = numpy.random.default_rng().normal(
            loc=0, scale=0.25, size=(number_of_outputs, number_of_inputs)
        )
        # The number of weights is equal to the number of weights
        self.weighted_velocities = numpy.zeros((number_of_outputs, number_of_inputs))
        self.number_of_inputs = number_of_inputs
        self.number_of_outputs = number_of_outputs
        self.activation_function = activation_function
        self.input_values = []

    @property
    def activation_values(self):
        """
        Take the weights times inputs with the bias added to get the outputs (i.e. inputs for the next node)
         inputs (j,1)  *         weights (n,j)         +       bias       =   outputs (n,1)
         Γ          Ꞁ      Γ                       Ꞁ        Γ         Ꞁ       Γ         Ꞁ
              i_11           w_1_1 w_2_1 ... w_j_1             b_11              z_11
              i_12     *     w_2_2 w_2_2 ... w_j_2     +       b_12       =      z_12
              ....                 ...                         ....              ....
              i_1j           w_n_1 w_n_2 ... w_j_n             b_1j              z_1n
         L           ⅃    L                       ⅃        L          ⅃       L         ⅃
        """
        return numpy.matmul(self.weights, self.input_values) + self.bias

    @property
    def activation_energies(self):
        """
        Pass the activation values through activation function to get activation energies
        """
        activation_function_vector = numpy.vectorize(self.activation_function.equation)
        return activation_function_vector(self.activation_values)

    @property
    def activation_function_derivatives(self):
        """
        Get the derivative value of the activation function with respect to the activation energy
        """
        activation_function_derivative_vector = numpy.vectorize(
            self.activation_function.derivative
        )
        return activation_function_derivative_vector(self.activation_values)

    def update_weights(self, decay):
        self.weights = (self.weights * decay) + self.weighted_velocities

    def update_biases(self, decay):
        self.bias = (self.bias * decay) + self.bias_velocities

    def reset_velocities(self):
        self.weighted_velocities = numpy.zeros(
            (self.number_of_outputs, self.number_of_inputs)
        )
        self.bias_velocities = numpy.zeros((self.number_of_outputs))
