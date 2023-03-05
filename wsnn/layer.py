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

        self.clear_gradients()
        self._inputs = [0] * num_nodes_in
        self._weighted_inputs = [0] * num_nodes_out
        self._activations = [0] * num_nodes_out

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
    def node_cost(output_activation: float, expected_output: float) -> float:
        error = output_activation - expected_output
        return error * error

    @staticmethod
    def node_cost_derivative(output_activation: float, expected_output: float):
        return 2 * (output_activation - expected_output)

    @staticmethod
    def activation_function(weighted_input: float) -> float:
        return 1 / (1 + math.exp(-weighted_input))

    @staticmethod
    def activation_function_derivative(weighted_input: float) -> float:
        activation = Layer.activation_function(weighted_input)
        return activation * (1 - activation)

    def calculate_output_layer_node_values(
        self, expected_outputs: Sequence[float]
    ) -> list[float]:
        node_values: list[float] = []

        for i in range(len(expected_outputs)):
            cost_derivative = Layer.node_cost_derivative(
                self._activations[i], expected_outputs[i]
            )
            activation_derivative = Layer.activation_function_derivative(
                self._weighted_inputs[i]
            )
            node_values.append(activation_derivative * cost_derivative)

        return node_values

    def calculate_hidden_layer_node_values(
        self, old_layer: "Layer", old_node_values: Sequence[float]
    ):
        new_node_values: list[float] = []

        for new_node_index in range(self.num_nodes_out):
            new_node_value = 0
            for old_node_index in range(len(old_node_values)):
                weighted_input_derivative = old_layer.weights[
                    new_node_index][old_node_index]
                new_node_value += (
                    weighted_input_derivative * old_node_values[old_node_index]
                )

            new_node_value *= self.activation_function_derivative(
                self._weighted_inputs[new_node_index]
            )
            new_node_values.append(new_node_value)
        return new_node_values

    def calculate_outputs(self, inputs: Sequence[float]) -> list[float]:
        self._inputs = inputs
        for node_out in range(self.num_nodes_out):
            weighted_input = self.biases[node_out]
            for node_in in range(self.num_nodes_in):
                weighted_input += inputs[node_in] * self.weights[node_in][node_out]
            self._weighted_inputs[node_out] = weighted_input
            self._activations[node_out] = Layer.activation_function(weighted_input)
        return self._activations.copy()

    def update_gradients(self, node_values: Sequence[float]) -> None:
        for node_out in range(self.num_nodes_out):
            for node_in in range(self.num_nodes_in):
                derivative_cost_wrt_weight = (
                    self._inputs[node_in] * node_values[node_out]
                )  # Derivative of cost with respect to weight
                self.cost_gradients_W[node_in][node_out] += derivative_cost_wrt_weight

            derivative_cost_wrt_bias = (
                1 * node_values[node_out]
            )  # Derivative of cost with respect to bias
            self.cost_gradients_B[node_out] += derivative_cost_wrt_bias

    def apply_gradients(self, learn_rate: float) -> None:
        for node_out in range(self.num_nodes_out):
            self.biases[node_out] -= self.cost_gradients_B[node_out] * learn_rate
            for node_in in range(self.num_nodes_in):
                
                self.weights[node_in][node_out] -= (
                    self.cost_gradients_W[node_in][node_out] * learn_rate
                )

    def clear_gradients(self):
        self.cost_gradients_W = [[0.0] * self.num_nodes_out for _ in range(self.num_nodes_in)]
        self.cost_gradients_B = [0.0] * self.num_nodes_out

    def initalize_random_weights(self) -> None:
        for node_out in range(self.num_nodes_out):
            for node_in in range(self.num_nodes_in):
                random_value = random.uniform(-1, 1)
                self.weights[node_in][node_out] = random_value / self.num_nodes_in ** (
                    1 / 2
                )
