import logging
import numpy
from pprint import pprint

logger = logging.getLogger("neural_network.log")


def evolve(
    layers,
    cost_function,
    output,
    expected_output,
    regularization,
    learn_rate,
    activation_function_derivatives,
):
    weightDecay = 1 - regularization * learn_rate
    # Calculate the gradient of the cost with respect to the outputs of the network
    output_cost_gradient = calculate_cost_gradient(
        cost_function, output, expected_output
    )

    # Calculate the gradient of the cost with respect to the biases of the network
    deriv_cost_wrt_afo = numpy.array([0, 0, output_cost_gradient[-1]])
    deriv_af_wrt_av = activation_function_derivatives[0][1]
    deriv_av_wrt_w = numpy.array(
        [[0, 0], [0, 0], [layers[-1].weighted_inputs[-1][-1], 0]]
    )
    logger.debug(
        f"""
    weightDecay: {weightDecay}
    output_cost_gradient: {output_cost_gradient}
    activation_function_derivatives: {activation_function_derivatives}
    derivative of cost with respect to activation function output: {deriv_cost_wrt_afo}
    derivative of activation function with respect to activation value: {deriv_af_wrt_av}
    derivative of activation value with respect to weight: {deriv_av_wrt_w}
    test: {deriv_cost_wrt_afo*deriv_af_wrt_av}
    """
    )


def calculate_cost_gradient(
    cost_function,
    output,
    expected_output,
):
    """
    Calculate gradient of cost with respect to activation outputs
     δC     δC        δC
    ---- , ----, ... ----
    δa_1   δa_2      δa_n
    """
    return cost_function.derivative(output, expected_output)
