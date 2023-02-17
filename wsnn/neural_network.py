from typing import Sequence
from wsnn.data import DataPoint
from wsnn.layer import Layer
import pickle


class NeuralNetwork(object):
    def __init__(self, layer_sizes: Sequence[int]) -> None:
        self.layers: list[Layer] = []
        self.lowest_cost = 1
        for i in range(len(layer_sizes) - 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))

    def calculate_outputs(self, inputs: Sequence[float]) -> Sequence[float]:
        for layer in self.layers:
            inputs = layer.calculate_outputs(inputs)
        return inputs

    def cost(self, data_point: DataPoint) -> float:
        outputs = self.calculate_outputs(data_point.inputs)
        cost = 0.0

        for node_out in range(len(outputs)):
            cost += Layer.node_cost(
                outputs[node_out], data_point.expected_outputs[node_out]
            )

        return cost

    def cost_multiple(self, data: Sequence[DataPoint]) -> float:
        total_cost = 0.0

        for data_point in data:
            total_cost += self.cost(data_point)

        return total_cost / len(data)

    def classify(self, inputs):
        outputs = self.calculate_outputs(inputs)
        return outputs.index(max(outputs))

    def apply_all_gradients(self, learn_rate: float):
        for layer in self.layers:
            layer.apply_gradients(learn_rate)

    def save_weights_and_biases(self, cost: float):
        if cost < self.lowest_cost:
            with open("train.pkl", "wb") as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            self.lowest_cost = cost

    def learn(self, training_data: Sequence[DataPoint], learn_rate: float):
        h = 0.0001
        original_cost = self.cost_multiple(training_data)
        for layer in self.layers:
            for node_out in range(layer.num_nodes_out):
                for node_in in range(layer.num_nodes_in):
                    layer.weights[node_in][node_out] += h
                    delta_cost = self.cost_multiple(training_data) - original_cost
                    layer.weights[node_in][node_out] -= h
                    layer.cost_gradients_W[node_in][node_out] = delta_cost / h

            for bias_index in range(len(layer.biases)):
                layer.biases[bias_index] += h
                delta_cost = self.cost_multiple(training_data) - original_cost
                layer.biases[bias_index] -= h
                layer.cost_gradients_B[bias_index] = delta_cost / h

        self.apply_all_gradients(learn_rate)
