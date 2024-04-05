import numpy
from pandas import DataFrame


class CostFunction:
    def __init__(self) -> None:
        pass


class MeanAbsolute(CostFunction):
    def calculate_error(predicted_value: DataFrame, actual_value: DataFrame):
        return numpy.sum(predicted_value - actual_value)

    def derivative(predicted_value: DataFrame, actual_value: DataFrame):
        return 1


class MeanSquared(CostFunction):
    def calculate_error(predicted_value, actual_value):
        error = predicted_value - actual_value
        length = error.shape[0]
        return numpy.sum(error**2) / length

    def derivative(predicted_value: DataFrame, actual_value: DataFrame):
        return 2 * (predicted_value - actual_value)


class Huber(CostFunction):
    def calculate_error(
        predicted_value: DataFrame, actual_value: DataFrame, delta: float
    ):
        error = predicted_value - actual_value
        if abs(error) > delta:
            return 0.5 * (error * error)
        return numpy.sum(delta * (error) - 0.5 * delta * delta)

    def derivative(predicted_value: DataFrame, actual_value: DataFrame, delta: float):
        error = predicted_value - actual_value
        if abs(error) > delta:
            return error * error
        return delta


cost_function_options = {
    "meanabs": MeanAbsolute,
    "meansq": MeanSquared,
    "huber": Huber,
}
