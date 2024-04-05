import activation_functions
import argparse
import cost
from create_logger import create_logger
import json
from layer import Layer
import pandas
import numpy
import sys
import os
import matplotlib.pyplot as plt

from neural_network.evolve import Evolve


# A = activation function
# i = activation energy
# z = activation value
# w = weight

logger = create_logger(
    name="neural_network",
    loggerLevel="DEBUG",
    fileHandlerLevel="DEBUG",
    consoleHandlerLevel="INFO",
)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        if isinstance(obj, numpy.floating):
            return float(obj)
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


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
    number_of_nodes_per_layer: {parameters["number_of_nodes_per_layer"]}
    name: {parameters["name"]} 
    actual_value_key: {parameters["actual_value_key"]}
    parameters_file: {args.parameters_file}
    percent_of_data: {parameters["percentage_of_data"]}
    """
    )

    output_df = input_df[parameters["actual_value_key"]]
    unique_outputs = list(set(output_df))
    unique_outputs.sort()
    logger.debug(f"Discrete outputs: {unique_outputs}")

    # Create each layer
    layers = []
    # Number of layers = number of hidden layers + output layer
    number_of_layers = len(parameters["number_of_nodes_per_layer"]) + 1
    for layer_index in range(0, number_of_layers, 1):
        layers.append(
            Layer(
                name=(
                    f"Layer {layer_index+1}"
                    if layer_index < number_of_layers - 1
                    else "Output Layer"
                ),
                number_of_inputs=(
                    # take the data frame columns, minus the outputs for the first layer
                    len(input_df.columns) - 1
                    if layer_index == 0
                    # Otherwise, the number of inputs should be the number of outputs from the last layer
                    else parameters["number_of_nodes_per_layer"][layer_index - 1]
                ),
                number_of_outputs=(
                    # The number of outputs from our number_of_nodes_per_layer parameters
                    len(unique_outputs)
                    if layer_index == number_of_layers - 1
                    # Unless we're at the last layer, then it should match the number of outputs
                    else parameters["number_of_nodes_per_layer"][layer_index]
                ),
                # Give the activation function
                activation_function=activation_functions.activation_function_options[
                    parameters["activation_function"]
                ],
            )
        )

    learn_rate = parameters["initial_learning_rate"]
    evolve = Evolve(
        layers=layers,
        cost_function=cost.cost_function_options[parameters["cost_function"]],
        regularization=parameters["regularization_strength"],
        learn_rate=learn_rate,
        momentum=parameters["momentum"],
    )

    # Get length of test data to use for training
    input_df_size, _ = input_df.shape
    # Get how large the training data is based on the percentage of data specified in the parameters
    training_data_length = int((input_df_size * parameters["percentage_of_data"]) / 100)
    # Get how many epochs we need based on the epoch size specified in the parameters
    number_of_epochs = int(training_data_length / parameters["epoch_size"])
    logger.debug(
        f"""
    traning_data_length: {training_data_length}
    number_of_epochs: {number_of_epochs}
    """
    )
    if not number_of_epochs:
        number_of_epochs = 1
    # Convert the input data to a numpy array
    training_data_from_df = input_df.to_numpy()
    # Slice off the correct amount of training data
    training_data = training_data_from_df[:training_data_length, :]
    # Randomize the training data
    numpy.random.shuffle(training_data)
    # Split the training data into the correct amount of epochs
    epochs = numpy.array_split(
        training_data,
        number_of_epochs,
    )

    epoch_plot_data = []
    error_plot_data = []
    # Iterate over the epochs
    for epoch_index, epoch in enumerate(epochs):
        epoch_plot = [[], []]
        error_plot = []
        epoch_cost_total = 0
        logger.debug(f"--------EPOCH {epoch_index+1} START--------")
        # Iterate over the data points in the epoch
        for data_index, data in enumerate(epoch):
            for layer_index, layer in enumerate(layers):
                if layer_index == 0:
                    # If first layer, input the data
                    layer.input_values = data[:-1]
                else:
                    # Pass outputs of previous layers to next layer
                    layer.input_values = layers[layer_index - 1].activation_energies
                # layer.print_layer()

            # Get the column with the number closest to 1
            column_number = numpy.where(
                layer.activation_energies
                == numpy.max(layer.activation_energies, axis=0)
            )[0][0]
            # Create an array of expected output
            expected_outputs = numpy.array([0 for output in unique_outputs])
            expected_outputs[unique_outputs.index(data[-1])] = 1
            # Calculate total cost
            epoch_cost = cost.cost_function_options[
                parameters["cost_function"]
            ].calculate_error(layer.activation_energies, expected_outputs)
            epoch_cost_total += epoch_cost
            error_plot.append(epoch_cost)
            epoch_plot[0].append(data[-1])
            epoch_plot[1].append(unique_outputs[column_number])
            logger.debug(
                f"""
            ========Data Point {data_index+1}===========
            Expected Outputs: {expected_outputs}
            Output of layer: {layer.activation_energies}
            Actual: {data[-1]}
            Predicted: {unique_outputs[column_number]}
            Unique Outputs: {unique_outputs}
            Epoch Cost Total: {epoch_cost_total}
            Epoch Cost Average: {epoch_cost_total/(data_index+1)}
            """
            )

            evolve.update_weight_velocities(list(data[:-1]), expected_outputs)
            # remove
            # break

        learn_rate = (
            1.0 / (1.0 + (parameters["learning_rate_decay"] * (epoch_index + 1)))
        ) * learn_rate
        evolve.learn_rate = learn_rate
        logger.debug(f"learn_rate: {learn_rate}")
        epoch_plot_data.append(epoch_plot)
        error_plot_data.append(error_plot)
        logger.debug(f"--------EPOCH {epoch_index+1} COMPLETE--------")
        # for layer in layers:
        #     layer.reset_velocities()
        # remove
        # break

    output_folder = f"./neural_network/trained_data/{parameters['name']}"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    with open(f"{output_folder}/outputs.json", "w") as f:
        json.dump(epoch_plot_data, f)
    with open(f"{output_folder}/errors.json", "w") as f:
        json.dump(error_plot_data, f, cls=NpEncoder)

    fig, axs = plt.subplots(number_of_epochs, 2, sharex=False, sharey=False)
    fig.suptitle("Data Fit")
    for plot_index, plot_datum in enumerate(epoch_plot_data):
        t = numpy.arange(0.0, len(plot_datum[0]), 1)
        axs[plot_index, 0].plot(t, plot_datum[0], "r--", t, plot_datum[1], "b--")
    for error_plot_index, error_plot_datum in enumerate(error_plot_data):
        t = numpy.arange(0.0, len(error_plot_datum), 1)
        axs[error_plot_index, 1].plot(t, error_plot_datum, "r--")
    axs[0, 0].set_title("Prediction vs Actual")
    axs[0, 1].set_title("Errors")
    plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
