import numpy as np
from sklearn.model_selection import train_test_split
from Organism import *
from DataGenerator import DataGenerator


def main():
    X_train, y_train, X_test, y_test = DataGenerator()
    # Create a List of all active GeneticNeuralNetworks
    networks = []
    pool = []
    # Track Generations
    generation = 0
    # Initial Population
    n = 20

    # Generate n randomly weighted neural networks
    for i in range(0, n):
        networks.append(Organism())

    # Cache Max Fitness
    max_fitness = 0

    # Max Fitness Weights
    optimal_weights = []

    # Evolution Loop
    while max_fitness < .9:
        # Log the current generation
        generation += 1
        print('Generation: ', generation)

        # Forward propagate the neural networks to compute a fitness score
        for nn in networks:
            # Propagate to calculate fitness score
            nn.forward_propagation(X_train, y_train)
            # Add to pool after calculating fitness
            pool.append(nn)

        # Clear for propagation of next children
        networks.clear()

        # Sort based on fitness
        pool = sorted(pool, key=lambda x: x.fitness)
        pool.reverse()

        # Find Max Fitness and Log Associated Weights
        for i in range(0, len(pool)):
            # If there is a new max fitness among the population
            if pool[i].fitness > max_fitness:
                max_fitness = pool[i].fitness
                print('Max Fitness: ', max_fitness)
                # Reset optimal_weights
                optimal_weights = []
                # Iterate through layers, get weights, and append to optimal
                for layer in pool[i].layers:
                    optimal_weights.append(layer.get_weights()[0])
                print(optimal_weights)

        # Crossover, top 5 randomly select 2 partners for child
        for i in range(0, 5):
            for j in range(0, 2):
                # Create a child and add to networks
                temp = dynamic_crossover(pool[i], random.choice(pool))
                # Add to networks to calculate fitness score next iteration
                networks.append(temp)

    # Create a Genetic Neural Network with optimal initial weights
    gnn = Organism(optimal_weights)
    gnn.compile_train(10, X_train, y_train)

    # Test the Genetic Neural Network Out of Sample
    y_hat = gnn.predict(X_test.values)
    print('Test Accuracy: %.2f' % accuracy_score(y_test, y_hat.round()))


if __name__ == '__main__':
    print("runing")
    main()
