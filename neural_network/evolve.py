import logging

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
                    # print(f"C/w_{layer_index+1}{node_index+1}{weight_index+1}")
                    weight_velocity = self.calculate_velocitiy(
                        layer_index=layer_index,
                        node_index=node_index,
                        weight_index=weight_index,
                        number_of_layers=len(self.layers),
                        calculate_type="weights",
                        depth=1,
                    )
                    # old velocity * momentum - what we just calculated * learn rate
                    # weights
                    current_weight_velocity = layer.weighted_velocities[
                        node_index, weight_index
                    ]
                    updated_weight_velocity = (
                        current_weight_velocity * self.momentum
                    ) - (weight_velocity * self.learn_rate)
                    layer.weighted_velocities[node_index, weight_index] = (
                        updated_weight_velocity
                    )

                # print(f"C/b_{layer_index+1}{node_index+1}")
                bias_velocity = self.calculate_velocitiy(
                    layer_index=layer_index,
                    node_index=node_index,
                    weight_index=weight_index,
                    number_of_layers=len(self.layers),
                    calculate_type="biases",
                    depth=1,
                )

                # biases
                current_bias_velocity = layer.bias_velocities[node_index]
                updated_bias_velocity = (current_bias_velocity * self.momentum) - (
                    bias_velocity * self.learn_rate
                )
                layer.bias_velocities[node_index] = updated_bias_velocity
            layer.update_biases(self.weight_decay)
            layer.update_weights(self.weight_decay)

    def calculate_velocitiy(
        self,
        layer_index,
        node_index,
        weight_index,
        number_of_layers,
        calculate_type,
        depth,
    ):
        # print("-------------------------------------")
        # print(layer_index, node_index, weight_index)
        cumulative_total = 0
        # If we're getting the derivative of the the last layer weights
        if layer_index == number_of_layers - 1:
            # print(f"{'  '*depth}C/a_{layer_index+1}{node_index+1}")
            return self.output_cost_gradient[
                node_index
            ] * self.recursive_layer_calculation(
                current_layer=number_of_layers - 1,
                current_node=node_index + 1,
                layer_index=layer_index,
                node_index=node_index,
                weight_index=weight_index,
                calculate_type=calculate_type,
                depth=depth,
            )
        else:
            for cost_index, cost_derivative in enumerate(self.output_cost_gradient):
                # print(f"  C/a_{number_of_layers}{cost_index+1}")
                cumulative_total += cost_derivative * self.recursive_layer_calculation(
                    current_layer=number_of_layers - 1,
                    current_node=cost_index,
                    layer_index=layer_index,
                    node_index=node_index,
                    weight_index=weight_index,
                    calculate_type=calculate_type,
                    depth=depth,
                )
            return cumulative_total

    def recursive_layer_calculation(
        self,
        current_layer,
        current_node,
        layer_index,
        node_index,
        weight_index,
        calculate_type,
        depth,
    ):
        if current_layer == layer_index:
            if calculate_type == "weights":
                # print(
                #     f"{'     '*(depth)}a_{layer_index+1}{node_index+1}/z_{layer_index+1}{node_index+1}*z_{layer_index+1}{node_index+1}/w_{layer_index+1}{node_index+1}{weight_index+1}"
                # )
                return (
                    self.layers[layer_index].activation_function_derivatives[node_index]
                    * self.layers[layer_index].input_values[weight_index]
                )
            elif calculate_type == "biases":
                # print(
                #     f"{'     '*(depth)}a_{layer_index+1}{node_index+1}/z_{layer_index+1}{node_index+1}*z_{layer_index+1}{node_index+1}/b_{layer_index+1}{node_index+1}"
                # )
                return (
                    self.layers[layer_index].activation_function_derivatives[node_index]
                    * self.layers[layer_index].bias[node_index]
                )

        total = 0
        # print(
        #     f"{'    '*(depth)}+ a_{current_layer+1}{current_node+1}/z_{current_layer+1}{current_node+1}"
        # )
        for internal_node_index in range(
            0, self.layers[current_layer].number_of_outputs, 1
        ):
            inner_total = 0

            for earlier_layer_node_index in range(
                0, self.layers[current_layer - 1].number_of_outputs, 1
            ):
                if (
                    current_layer == layer_index + 1
                    and earlier_layer_node_index != node_index
                ):
                    continue
                # print(
                #     f"{'     '*(depth)}+ z_{current_layer+1}{current_node+1}/a_{current_layer}{earlier_layer_node_index+1}"
                # )
                inner_total += self.recursive_layer_calculation(
                    current_layer - 1,
                    earlier_layer_node_index,
                    layer_index,
                    node_index,
                    weight_index,
                    calculate_type,
                    depth=depth + 2,
                )
                return (
                    inner_total
                    * self.layers[current_layer].weights[internal_node_index][
                        earlier_layer_node_index
                    ]
                )
            if internal_node_index == node_index or current_layer > layer_index:
                break
        return (
            total
            * self.layers[current_layer].activation_function_derivatives[current_node]
        )
