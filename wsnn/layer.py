from typing import Optional, Sequence
import math
import random


class Layer(object):
    def __init__(
        self,
        num_nodes_in: int,
        num_nodes_out: int,
        weights: Optional[list[list[float]]] = None,
        biases: Optional[list[float]] = None,
    ):
        self.num_nodes_in = num_nodes_in
        self.num_nodes_out = num_nodes_out
        self.cost_gradients_W = [[0.0] * num_nodes_out for _ in range(num_nodes_in)]
        self.cost_gradients_B = [0.0] * num_nodes_out
        exceptions = []

        if not weights:
            self._weights = [[0.0] * num_nodes_out for _ in range(num_nodes_in)]
        else:
            try:
                self.weights = weights
            except ValueError as e:
                exceptions.append(e)
        if not biases:
            self._biases = [0.0] * num_nodes_out
        else:
            try:
                self.biases = biases
            except ValueError as e:
                exceptions.append(e)
        if exceptions:
            raise Exception(*exceptions)

    @property
    def weights(self) -> list[list[float]]:
        return self._weights

    @weights.setter
    def weights(self, v: list[list[float]]) -> None:
        if len(v) != self.num_nodes_in:
            raise ValueError("Number of weight sets must match num_nodes_in")
        if len([x for x in v if len(x) != self.num_nodes_out]):
            raise ValueError("Number of weights must match num_nodes_out")
        self._weights = v

    @property
    def biases(self) -> list[float]:
        return self._biases

    @biases.setter
    def biases(self, v: list[float]) -> None:
        if len(v) != self.num_nodes_out:
            raise ValueError("Number of bias inputs must match num_nodes_out")
        self._biases = v

    @staticmethod
    def node_cost(output_activation: float, expected_outputs: float) -> float:
        error = output_activation - expected_outputs
        return error * error

    @staticmethod
    def activation_function(weighted_input: float) -> float:
        return 1 / (1 + math.exp(weighted_input))

    def calculate_outputs(self, inputs: Sequence[float]) -> list[float]:
        activations: list[float] = []
        for node_out in range(self.num_nodes_out):
            weighted_input = self.biases[node_out]
            for node_in in range(self.num_nodes_in):
                weighted_input += inputs[node_in] * self.weights[node_in][node_out]
            activations.append(Layer.activation_function(weighted_input))
        return activations

    def apply_gradients(self, learn_rate: float):
        for node_out in range(self.num_nodes_out):
            self.biases[node_out] -= self.cost_gradients_B[node_out] * learn_rate
            for node_in in range(self.num_nodes_in):
                self.weights[node_in][node_out] -= (
                    self.cost_gradients_W[node_in][node_out] * learn_rate
                )

    def initalize_random_weights(self):
        for node_out in range(self.num_nodes_out):
            for node_in in range(self.num_nodes_in):
                random_value = random.uniform(-1, 1)
                self.weights[node_in][node_out] = random_value / self.num_nodes_in ** (
                    1 / 2
                )
