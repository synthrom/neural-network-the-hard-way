import logging
from pprint import pprint

logger = logging.getLogger("neural_network.log")


class Evolve:
    """
    A = activation function
    α = activation energy
    z = activation value
    w = weight
    """

    def __init__(
        self,
        layers,
        cost_function,
        regularization,
        learn_rate,
        momentum,
    ) -> None:
        self.layers = layers
        self.initial_inputs = []
        self.cost_function = cost_function
        self.expected_output = []
        self.regularization = regularization
        self.learn_rate = learn_rate
        self.weight_decay = 1 - self.regularization * self.learn_rate
        self.momentum = momentum

    @property
    def output_cost_gradient(self):
        """
        Calculate gradient of cost with respect to activation outputs
        δC     δC        δC
        ---- , ----, ... ----
        δa_1   δa_2      δa_n
        """
        return self.cost_function.derivative(
            self.layers[-1].activation_energies,
            self.expected_output,
        )

    def update_weight_velocities(self, initial_inputs, expected_output):
        self.initial_inputs = initial_inputs
        self.expected_output = expected_output

        # Calculate the weight velocities by iterating over the layers, nodes, and weight velocities
        for layer_index, layer in enumerate(self.layers):
            number_of_nodes, number_of_weights = layer.weighted_velocities.shape
            for node_index in range(0, number_of_nodes, 1):
                for weight_index in range(0, number_of_weights, 1):
                    self.calculate_velocitiy(
                        layer_index=layer_index,
                        node_index=node_index,
                        weight_index=weight_index,
                        number_of_layers=len(self.layers),
                        calculate_type="weights",
                    )
                self.calculate_velocitiy(
                    layer_index=layer_index,
                    node_index=node_index,
                    weight_index=weight_index,
                    number_of_layers=len(self.layers),
                    calculate_type="biases",
                )
            self.layers[layer_index].update_biases(self.weight_decay)
            self.layers[layer_index].update_weights(self.weight_decay)

    def calculate_velocitiy(
        self,
        layer_index,
        node_index,
        weight_index,
        number_of_layers,
        calculate_type,
    ):
        cumulative_total = 0
        # If we're getting the derivative of the the last layer weights
        if layer_index == number_of_layers - 1:

            cumulative_total = self.output_cost_gradient[
                node_index
            ] * self.recursive_layer_calculation(
                current_layer=number_of_layers - 1,
                layer_index=layer_index,
                node_index=node_index,
                weight_index=weight_index,
                calculate_type=calculate_type,
            )
        else:
            for cost_derivative in self.output_cost_gradient:
                cumulative_total += cost_derivative * self.recursive_layer_calculation(
                    current_layer=number_of_layers - 1,
                    layer_index=layer_index,
                    node_index=node_index,
                    weight_index=weight_index,
                    calculate_type=calculate_type,
                )
        # old velocity * momentum - what we just calculated * learn rate
        if calculate_type == "weights":
            current_weight_velocity = self.layers[layer_index].weighted_velocities[
                node_index, weight_index
            ]
            updated_velocity = (current_weight_velocity * self.momentum) - (
                cumulative_total * self.learn_rate
            )
            self.layers[layer_index].weighted_velocities[
                node_index, weight_index
            ] = updated_velocity
        elif calculate_type == "biases":
            current_bias_velocity = self.layers[layer_index].bias_velocities[node_index]
            updated_bias_velocity = (current_bias_velocity * self.momentum) - (
                cumulative_total * self.learn_rate
            )
            self.layers[layer_index].bias_velocities[node_index] = updated_bias_velocity

    def recursive_layer_calculation(
        self,
        current_layer,
        layer_index,
        node_index,
        weight_index,
        calculate_type,
    ):
        if current_layer == layer_index:
            if calculate_type == "weights":
                return (
                    self.layers[layer_index].activation_function_derivatives[node_index]
                    * self.layers[layer_index].input_values[weight_index]
                )
            elif calculate_type == "biases":
                return (
                    self.layers[layer_index].activation_function_derivatives[node_index]
                    * self.layers[layer_index].bias[node_index]
                )
        total = 0
        for internal_node_index in range(0, self.layers[layer_index].number_of_outputs):
            inner_total = 0
            for earlier_layer_node_index in range(
                self.layers[current_layer - 1].number_of_outputs - 1, -1, -1
            ):
                inner_total += self.recursive_layer_calculation(
                    current_layer - 1,
                    layer_index,
                    node_index,
                    weight_index,
                    calculate_type,
                )
            total += (
                inner_total
                * self.layers[layer_index].weights[internal_node_index][weight_index]
            )
        return (
            total * self.layers[layer_index].activation_function_derivatives[node_index]
        )
