import math
import random

# Deep Neural Network (DNN) Implementation from scratch
# Focusing on foundational understanding and precision
# 深層学習 (Deep Learning) / ディープニューラルネットワーク (DNN)

class DenseLayer:
    def __init__(self, input_size, output_size, activation='tanh'):
        # Initialize weights with small random values
        self.weights = [[random.uniform(-0.1, 0.1) for _ in range(input_size)] for _ in range(output_size)]
        self.biases = [0.0 for _ in range(output_size)]
        self.activation = activation

    def activate(self, x):
        if self.activation == 'tanh':
            return math.tanh(x)
        elif self.activation == 'relu':
            return max(0, x)
        elif self.activation == 'sigmoid':
            try:
                return 1 / (1 + math.exp(-x))
            except OverflowError:
                return 0.0 if x < 0 else 1.0
        return x

    def forward(self, inputs):
        self.inputs = inputs
        self.output = []
        for i in range(len(self.weights)):
            # Dot product: sum(w * x) + b
            sum_val = sum(self.weights[i][j] * inputs[j] for j in range(len(inputs)))
            sum_val += self.biases[i]
            self.output.append(self.activate(sum_val))
        return self.output

class DeepNeuralNetwork:
    def __init__(self, layers_config):
        """
        layers_config: list of integers representing nodes in each layer
        e.g., [3, 5, 2] means 3 inputs, 5 hidden nodes, 2 outputs
        """
        self.layers = []
        for i in range(len(layers_config) - 1):
            self.layers.append(DenseLayer(layers_config[i], layers_config[i+1]))

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

def main():
    print("====================================================")
    print("  Deep Neural Network (DNN) - Implementation From Scratch")
    print("  深層学習 (Deep Learning) / ディープニューラルネットワーク")
    print("====================================================")
    
    # Example Configuration:
    # Input Layer: 4
    # Hidden Layer 1: 8
    # Hidden Layer 2: 8
    # Output Layer: 3
    config = [4, 8, 8, 3]
    dnn = DeepNeuralNetwork(config)
    
    # Sample training-like data (Normalized)
    sample_input = [0.1, 0.5, -0.3, 0.8]
    print(f"Input Vector:  {sample_input}")
    
    # Perform forward propagation
    output = dnn.forward(sample_input)
    
    print("-" * 52)
    print(f"DNN Output:    {output}")
    print("====================================================")

if __name__ == "__main__":
    main()
