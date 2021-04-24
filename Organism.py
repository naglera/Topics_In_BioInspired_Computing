import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Dense


# Could make a generalized instance of the Model Keras class for a GNN
# New Type of Neural Network
class Organism(Sequential):
    # Constructor
    def __init__(self, child_weights=None):
        # Initialize Sequential Model Super Class
        super().__init__()
        # If no weights provided randomly generate them
        if child_weights is None:
            # Layers are created and randomly generated
            layer1 = Dense(11, input_shape=(11,), activation='sigmoid')
            layer2 = Dense(8, activation='sigmoid')
            layer3 = Dense(5, activation='sigmoid')
            layer4=Dense(1,activation='softmax')
            # Layers are added to the model
            self.add(layer1)
            self.add(layer2)
            self.add(layer3)
            self.add(layer4)
        # If weights are provided set them within the layers
        else:
            # Set weights within the layers
            self.add(
                Dense(
                    11,
                    input_shape=(11,),
                    activation='sigmoid',
                    weights=[child_weights[0], np.zeros(11)])
                )
            self.add(
                Dense(
                 8,
                 activation='sigmoid',
                 weights=[child_weights[1], np.zeros(8)])
            )
            self.add(
                Dense(
                 5,
                 activation='sigmoid',
                 weights=[child_weights[2], np.zeros(5)])
            )
            self.add(
                Dense(
                    1,
                    activation='softmax',
                    weights=[child_weights[3], np.zeros(1)])
            )

    # Function for forward propagating a row vector of a matrix
    def forward_propagation(self, X_train, y_train):
        # Forward propagation
        y_hat = self.predict(X_train.values)
        # Compute fitness score
        self.fitness = accuracy_score(y_train, y_hat.round())

    # Standard Backpropagation
    def compile_train(self, epochs,X_train,y_train):
        self.compile(
                      optimizer='rmsprop',
                      loss='binary_crossentropy',
                      metrics=['accuracy']
                      )
        self.fit(X_train.values, y_train.values, epochs=epochs)


# Chance to mutate weights
def mutation(child_weights):
    # Add a chance for random mutation
    selection = random.randint(0, len(child_weights)-1)
    mut = random.uniform(0, 1)
    if mut >= .5:
        child_weights[selection] *= random.randint(2, 5)
    else:
        # No mutation
        pass


# Crossover traits between two Genetic Neural Networks
def dynamic_crossover(nn1, nn2):
    # Lists for respective weights
    nn1_weights = []
    nn2_weights = []
    child_weights = []
    # Get all weights from all layers in the first network
    for layer in nn1.layers:
        nn1_weights.append(layer.get_weights()[0])

    # Get all weights from all layers in the second network
    for layer in nn2.layers:
        nn2_weights.append(layer.get_weights()[0])

    # Iterate through all weights from all layers for crossover
    for i in range(0, len(nn1_weights)):
        # Get single point to split the matrix in parents based on # of cols
        split = random.randint(0, np.shape(nn1_weights[i])[1]-1)
        # Iterate through after a single point and set the remaing cols to nn_2
        for j in range(split, np.shape(nn1_weights[i])[1]-1):
            nn1_weights[i][:, j] = nn2_weights[i][:, j]

        # After crossover add weights to child
        child_weights.append(nn1_weights[i])

    # Add a chance for mutation
    mutation(child_weights)

    # Create and return child object
    child = Organism(child_weights)
    return child

