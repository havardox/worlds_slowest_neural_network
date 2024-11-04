from typing import Sequence
from wsnn.data import DataPoint
from wsnn.layer import Layer
import pickle

class NeuralNetwork(object):
    def __init__(self, layer_sizes: Sequence[int]) -> None:
        # Initializes the neural network with layers based on specified layer sizes.
        self.layers: list[Layer] = []
        self.lowest_cost = 1
        for i in range(len(layer_sizes) - 1):
            # Create a layer between each consecutive layer size.
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i + 1]))

    def calculate_outputs(self, inputs: Sequence[float]) -> Sequence[float]:
        # Passes inputs through each layer to calculate final network output.
        for layer in self.layers:
            inputs = layer.calculate_outputs(inputs)
        return inputs

    def cost(self, data_point: DataPoint) -> float:
        # Computes total cost for a single data point using squared error.
        outputs = self.calculate_outputs(data_point.inputs)
        cost = 0.0
        for node_out in range(len(outputs)):
            cost += Layer.node_cost(
                outputs[node_out], data_point.expected_outputs[node_out]
            )
        return cost

    def cost_multiple(self, data: Sequence[DataPoint]) -> float:
        # Calculates average cost over multiple data points.
        total_cost = 0.0
        for data_point in data:
            total_cost += self.cost(data_point)
        return total_cost / len(data)

    def classify(self, inputs):
        # Classifies input by finding the index of the highest output value.
        outputs = self.calculate_outputs(inputs)
        return outputs.index(max(outputs))

    def apply_all_gradients(self, learn_rate: float):
        # Applies gradients to update weights and biases for all layers.
        for layer in self.layers:
            layer.apply_gradients(learn_rate)

    def save_weights_and_biases(self, cost: float):
        # Saves the network state if the current cost is the lowest observed.
        if cost < self.lowest_cost:
            with open("train.pkl", "wb") as f:
                pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
            self.lowest_cost = cost

    def learn(self, training_data: Sequence[DataPoint], learn_rate: float):
        h = 0.0001  # Small step for finite difference approximation of gradient
        original_cost = self.cost_multiple(training_data)  # Calculate initial cost

        # Loop through each layer to update weights and biases
        for layer in self.layers:
            # Loop through each output node in the layer
            for node_out in range(layer.num_nodes_out):
                # Loop through each input node connected to this output node
                for node_in in range(layer.num_nodes_in):
                    # Temporarily increase the weight by a small step h
                    layer.weights[node_in][node_out] += h
                    
                    # Recalculate the cost with this modified weight
                    delta_cost = self.cost_multiple(training_data) - original_cost
                    
                    # Restore the original weight
                    layer.weights[node_in][node_out] -= h
                    
                    # Calculate the gradient as the change in cost divided by h
                    # This approximates the partial derivative of cost with respect to this weight
                    layer.cost_gradients_W[node_in][node_out] = delta_cost / h

            # Repeat a similar process for biases in this layer
            for bias_index in range(len(layer.biases)):
                # Temporarily increase the bias by h
                layer.biases[bias_index] += h
                
                # Recalculate the cost with this modified bias
                delta_cost = self.cost_multiple(training_data) - original_cost
                
                # Restore the original bias
                layer.biases[bias_index] -= h
                
                # Calculate the gradient as the change in cost divided by h
                # This approximates the partial derivative of cost with respect to this bias
                layer.cost_gradients_B[bias_index] = delta_cost / h

        # Apply all the gradients computed for weights and biases with the learning rate
        self.apply_all_gradients(learn_rate)

