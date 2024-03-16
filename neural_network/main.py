import activation_functions
import argparse
import cost
from create_logger import create_logger
import json
from layer import Layer
import pandas
import numpy
import sys

from neural_network.evolve import evolve

logger = create_logger(
    name="neural_network",
    loggerLevel="DEBUG",
    fileHandlerLevel="DEBUG",
    consoleHandlerLevel="INFO",
)


def load_files(input_file: str) -> pandas.DataFrame:
    if "csv" in input_file:
        return pandas.read_csv(input_file)
    else:
        return pandas.read_json(input_file)


def main():
    """
    Training Code
    Uses back propogation to create neural network
    """
    parser = argparse.ArgumentParser()

    # JSON/CSV file that has inputs
    parser.add_argument(
        "-i", "--inputs", type=str, help="JSON or CSV file of input data", required=True
    )
    # Parameters file
    parser.add_argument(
        "-p",
        "--parameters-file",
        type=str,
        help="Path to parameters file",
        required=True,
    )
    args = parser.parse_args()

    # Load inputs and outputs
    input_df = load_files(args.inputs)
    with open(args.parameters_file, "r") as f:
        parameters = json.load(f)

    logger.info(
        f"""
    actiavtion_funciton: {parameters["activation_function"]} 
    coefficient: {parameters["coefficient"]}
    cost_function: {parameters["cost_function"]}
    epoch_size: {parameters["epoch_size"]}
    inputs: {args.inputs}
    hidden_layers: {parameters["hidden_layers"]}
    name: {parameters["name"]} 
    actual_value_key: {parameters["actual_value_key"]}
    parameters_file: {args.parameters_file}
    percent_of_data: {parameters["percentage_of_data"]}
    """
    )

    output_df = input_df[parameters["actual_value_key"]]
    unique_outputs = list(set(output_df))
    unique_outputs.sort()

    # Create each layer
    layers = []
    for index, node in enumerate(range(0, len(parameters["hidden_layers"]))):
        layers.append(
            Layer(
                # Name of layer
                name=f"Layer {index}",
                # Number of nodes
                nodes=parameters["hidden_layers"][node],
                # number of incoming inputs
                number_of_inputs=(
                    len(input_df.columns) - 1
                    if (index - 1) < 0
                    else parameters["hidden_layers"][index - 1]
                ),
                # Set the number of outputs based on all nodes; if the last node, set to number of desired outputs
                number_of_outputs=(
                    parameters["hidden_layers"][index + 1]
                    if (index + 1) < len(parameters["hidden_layers"])
                    else len(unique_outputs)
                ),
                # Give the activation function
                activation_function=activation_functions.activation_function_options[
                    parameters["activation_function"]
                ],
            )
        )
    # Add the output layer
    layers.append(
        Layer(
            # Name
            name="Output layer",
            # number of nodes
            nodes=len(unique_outputs),
            # number of incoming inputs
            number_of_inputs=(
                parameters["hidden_layers"][-1]
                if parameters["hidden_layers"]
                else len(input_df.columns) - 1
            ),
            # Set the number of outputs based on all nodes; if the last node, set to number of desired outputs
            number_of_outputs=len(unique_outputs),
            # Give the activation function
            activation_function=activation_functions.activation_function_options[
                parameters["activation_function"]
            ],
        )
    )

    # Get length of test data to use for training
    training_data_length = int((input_df.size * parameters["percentage_of_data"]) / 100)

    # Create array for capturing activation function derivative information
    activation_function_derivatives = []

    learn_rate = parameters["initial_learning_rate"]

    # Split up the data into epochs based on epoch size
    for epoch_index, epoch in enumerate(
        numpy.array_split(
            numpy.array(input_df[:training_data_length].sample(frac=1)),
            int(training_data_length / parameters["epoch_size"]),
        )
    ):
        epoch_cost_total = 0
        for data_index, data in enumerate(epoch):
            # TODO: derivative of activation values with respect to weight
            # TODO: derivative of activation values with respect to bias
            activation_function_derivative_layer = []
            # Pass initial input data through first layer
            # Calculate activation values z
            activation_values = (
                numpy.matmul(layers[0].weighted_inputs, data[:-1]) + layers[0].bias
            )
            # Pass activation values through activation function A(z)
            output_of_layer = layers[0].get_outputs(activation_values)
            # Get derivative of activation function with respect to values for use later
            activation_function_derivative_values = layers[0].get_derivative(
                activation_values
            )
            # Add derivative values to list
            activation_function_derivative_layer.append(
                activation_function_derivative_values
            )
            for layer in layers[1:]:
                # Pass outputs of previous layers to next layer
                # Do same steps as above (calculate z, pass to activation function, get derivative values)
                activation_values = (
                    numpy.matmul(layer.weighted_inputs, output_of_layer) + layer.bias
                )
                output_of_layer = layer.get_outputs(activation_values)
                activation_function_derivative_values = layer.get_derivative(
                    activation_values
                )
                activation_function_derivative_layer.append(
                    activation_function_derivative_values
                )
            # Add the derivatives to the list of derivatives of the activation funciton
            activation_function_derivatives.append(activation_function_derivative_layer)
            logger.debug(
                f"activation_function_derivative_layer: {activation_function_derivative_layer}"
            )
            # Get the column with the number closest to 1
            column_number = numpy.where(
                output_of_layer == numpy.max(output_of_layer, axis=0)
            )[0][0]
            # Create an array of expected output
            expected_outputs = numpy.array([0 for output in unique_outputs])
            expected_outputs[unique_outputs.index(data[-1])] = 1
            # Calculate total cost
            epoch_cost_total += cost.cost_function_options[
                parameters["cost_function"]
            ].calculate_error(output_of_layer, expected_outputs)
            logger.debug(
                f"""
            ===================
            Expected Outputs: {expected_outputs}
            Output of layer: {output_of_layer}
            Actual: {data[-1]}
            Predicted: {unique_outputs[column_number]}
            Unique Outputs: {unique_outputs}
            Epoch Cost Total: {epoch_cost_total}
            Epoch Cost Average: {epoch_cost_total/(data_index+1)}
            """
            )
        # Use cost to evaluate change in variables
        evolve(
            layers=layers,
            cost_function=cost.cost_function_options[parameters["cost_function"]],
            output=output_of_layer,
            expected_output=expected_outputs,
            regularization=parameters["regularization_strength"],
            learn_rate=learn_rate,
            activation_function_derivatives=activation_function_derivatives,
        )
        learn_rate = (
            1.0 / (1.0 + parameters["learning_rate_decay"] * (epoch_index + 1))
        ) * parameters["initial_learning_rate"]
        logger.debug(f"learn_rate: {learn_rate}")
        logger.debug(f"--------EPOCH {epoch_index+1} COMPLETE--------")
    return 0


if __name__ == "__main__":
    sys.exit(main())
