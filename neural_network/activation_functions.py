from math import exp, atan, log, inf, pi


class ActivationFunction:
    def __init__(self) -> None:
        pass


class IdentityActivationFunction(ActivationFunction):
    def equation(x: float) -> float:
        return x

    def derivative(x: float) -> float:
        return 1


class BinaryStepActivationFunction(ActivationFunction):
    def equation(x: float) -> float:
        return 0 if x < 0 else 1

    def derivative(x: float) -> float:
        return 0 if x != 0 else 9999999


class LogisticActivationFunction(ActivationFunction):
    def equation(x: float) -> float:
        return 1 / (1 + exp(-1 * x))

    @classmethod
    def derivative(cls, x: float) -> float:
        return cls.equation(x) / (1 - cls.equation(x))


class TanhActivationFunction(ActivationFunction):
    def equation(x: float) -> float:
        try:
            return (2 / (1 + exp(-2 * x))) - 1
        except OverflowError:
            return -1

    @classmethod
    def derivative(cls, x: float) -> float:
        return 1 - (cls.equation(x) * cls.equation(x))


class ArctanActivationFunction(ActivationFunction):
    def equation(x: float) -> float:
        return atan(x)

    def derivative(x: float) -> float:
        return 1 / (x * x + 1)


class ReLUActivationFunction(ActivationFunction):
    def equation(x: float) -> float:
        return 0 if x < 0 else x

    def derivative(x: float) -> float:
        return 0 if x < 0 else 1


class PReLUActivationFunction(ActivationFunction):
    def equation(x: float, alpha: float) -> float:
        return alpha * x if x < 0 else x

    def derivative(x: float, alpha: float) -> float:
        return alpha if x < 0 else 1


class ELUActivationFunction(ActivationFunction):
    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = alpha

    def equation(self, x: float) -> float:
        return self.alpha * (exp(x) - 1) if x < 0 else x

    def derivative(self, x: float) -> float:
        return self.alpha * (exp(x) - 1) if x < 0 else 1


class SoftPlusActivationFunction(ActivationFunction):
    def equation(x: float) -> float:
        return log(1 + exp(x))

    def derivative(x: float) -> float:
        return 1 / (1 + exp(-1 * x))


class GaussianActivationFunction(ActivationFunction):
    def equation(x: float) -> float:
        return exp(-1 * x * x)

    def derivative(x: float) -> float:
        return -2 * x * exp(-1 * x * x)


activation_function_options = {
    "identity": IdentityActivationFunction,
    "binary": BinaryStepActivationFunction,
    "logistic": LogisticActivationFunction,
    "tanh": TanhActivationFunction,
    "arctan": ArctanActivationFunction,
    "ReLU": ReLUActivationFunction,
    "PReLU": PReLUActivationFunction,
    "ELU": ELUActivationFunction,
    "softPlus": SoftPlusActivationFunction,
    "gaussian": GaussianActivationFunction,
}
