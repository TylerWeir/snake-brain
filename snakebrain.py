"""A library containing an implementation of a basic neural
neural network for use in various machine learning problems.

Author: Tyler Weir
Date: Feb 23, 2022
"""

import numpy as np
from activations import *

class NeuralNet:
    """Represents a basic neural network."""

    def __init__(self, input_size, output_size, activation):
        # Fields to make reference easier
        self.input_size = input_size
        self.output_size = output_size

        # List of Numpy arrays to represent matrices holding 
        # weights of connections between nodes
        self.weights = [np.random.rand(input_size, output_size)]

        # List of Numpy arays representing vectors holding the 
        # activations of nodes
        self.nodes = [np.zeros(input_size), np.zeros(output_size)]

        # A list containing the activation function specified for 
        # each layer of nodes.  
        # Does the input layer use activations? TODO
        self.activations = [None, sigmoid]

    def add_hidden_layer(self, size, activation):
        # Add layer to nodes 
        self.nodes.insert(-1, np.zeros(size))

        # Add activation function
        self.activations.insert(-1, activation)
    
        # Remove the last entry of weights and add the two 
        # new entries
        self.weights.pop()

        # Weights into the hidden layer
        into_layer = len(self.nodes[-2])
        self.weights.append(np.random.rand(into_layer, size))

        # Weights out of the hidden layer
        self.weights.append(np.random.rand(size, self.output_size))

    def evaluate(self, data):
        """Passes `data` into the input layer of the net and returns 
        the activations of the output layer."""
        #TODO
        return None

    def train(self, num_epochs, learning_rate):
        """Trains the net for `num_epochs` at `learning_rate`."""
        #TODO
        return None

    def draw(self, canvas):
        """Draws a graph like representation of the current state of
        the neural network to the given tkinter canvas.  The gradient 
        represents the weights of the edges."""

        #HARDCODED VALUES RN TODO

        # First draw the nodes
        for i, _ in enumerate(self.nodes):
            for j, _ in enumerate(self.nodes[i]):
                x0 = i*20 + i*25
                y0 = j*20 + j*10
                x1 = x0 + 20
                y1 = y0 + 20
                canvas.create_oval(x0, y0, x1, y1, fill="green")

if __name__ == "__main__":
    skynet = NeuralNet(5, 3, relu)
    skynet.add_hidden_layer(6, relu)
    skynet.add_hidden_layer(4, relu)
    print(skynet.weights)

    # Import the require libs
    from tkinter import *
    
    # Make the window
    win=Tk()

    # Create a canvas widget
    canvas = Canvas(win, width=500, height=500)
    canvas.pack()

    # Add a line in canvas widget
    skynet.draw(canvas)

    win.mainloop()
