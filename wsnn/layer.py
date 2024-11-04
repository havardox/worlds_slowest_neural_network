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
        # Initialize layer with specified number of input and output nodes.
        # Optionally, custom weights and biases can be provided.
        self.num_nodes_in = num_nodes_in
        self.num_nodes_out = num_nodes_out
        # Gradients for cost with respect to weights and biases for backpropagation.
        self.cost_gradients_W = [[0.0] * num_nodes_out for _ in range(num_nodes_in)]
        self.cost_gradients_B = [0.0] * num_nodes_out
        exceptions = []

        # Initialize weights and biases, setting to zero if not provided.
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
        # Raise exceptions if any occurred during weight/bias setup.
        if exceptions:
            raise Exception(*exceptions)

    @property
    def weights(self) -> list[list[float]]:
        # Getter for weights with error checking.
        return self._weights

    @weights.setter
    def weights(self, v: list[list[float]]) -> None:
        # Setter for weights with dimension validation.
        if len(v) != self.num_nodes_in:
            raise ValueError("Number of weight sets must match num_nodes_in")
        if len([x for x in v if len(x) != self.num_nodes_out]):
            raise ValueError("Number of weights must match num_nodes_out")
        self._weights = v

    @property
    def biases(self) -> list[float]:
        # Getter for biases.
        return self._biases

    @biases.setter
    def biases(self, v: list[float]) -> None:
        # Setter for biases with length validation.
        if len(v) != self.num_nodes_out:
            raise ValueError("Number of bias inputs must match num_nodes_out")
        self._biases = v

    @staticmethod
    def node_cost(output_activation: float, expected_outputs: float) -> float:
        # Calculates squared error cost for an output node.
        error = output_activation - expected_outputs
        return error * error

    @staticmethod
    def activation_function(weighted_input: float) -> float:
        # Sigmoid activation function to calculate node output.
        return 1 / (1 + math.exp(weighted_input))

    def calculate_outputs(self, inputs: Sequence[float]) -> list[float]:
        # Calculates output activations for the layer based on input activations.
        activations: list[float] = []
        for node_out in range(self.num_nodes_out):
            # Calculate weighted input for each output node.
            weighted_input = self.biases[node_out]
            for node_in in range(self.num_nodes_in):
                weighted_input += inputs[node_in] * self.weights[node_in][node_out]
            # Apply activation function and store output.
            activations.append(Layer.activation_function(weighted_input))
        return activations

    def apply_gradients(self, learn_rate: float):
        # Adjust weights and biases based on calculated gradients.
        for node_out in range(self.num_nodes_out):
            self.biases[node_out] -= self.cost_gradients_B[node_out] * learn_rate
            for node_in in range(self.num_nodes_in):
                self.weights[node_in][node_out] -= (
                    self.cost_gradients_W[node_in][node_out] * learn_rate
                )

    def initalize_random_weights(self):
        # Initialize weights randomly to small values for symmetry breaking.
        for node_out in range(self.num_nodes_out):
            for node_in in range(self.num_nodes_in):
                random_value = random.uniform(-1, 1)
                self.weights[node_in][node_out] = random_value / self.num_nodes_in ** (
                    1 / 2
                )
