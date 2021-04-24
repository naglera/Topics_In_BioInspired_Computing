from Organism import *
from DataGenerator import DataGenerator

# params
dataset_name = 'iris.csv'
class_col = 'variety'
gen_size = 20
num_of_gens = 100


def main():
    X_train, Y_train, X_test, Y_test = DataGenerator(dataset_name, class_col)
    cur_gen = []
    prev_gen = []
    n_gen = 0
    best_org_fit = 0
    best_org_weights = []

    # initializing generation 0
    for i in range(0, gen_size):
        cur_gen.append(Organism())

    # start evolution
    while n_gen < num_of_gens:
        cur_best_fit = 0
        n_gen += 1

        # compute fitness for each organism on current generation
        for org in cur_gen:
            org.forward_propagation(X_train, Y_train)
            prev_gen.append(org)

        cur_gen.clear()
        prev_gen = sorted(prev_gen, key=lambda x: x.fitness)
        prev_gen.reverse()
        # save the fitness of the best organism of current generation
        cur_best_fit = prev_gen[0].fitness
        # save the fitness and weights of the best organism of until now
        if cur_best_fit > best_org_fit:
            best_org_fit = prev_gen[i].fitness
            best_org_weights.clear()
            for layer in prev_gen[i].layers:
                best_org_weights.append(layer.get_weights()[0])

        print('Generation: %1.f' % n_gen, '\tBest Fitness: %.4f' % cur_best_fit)

        # crossover
        for i in range(0, 5):
            for j in range(0, 2): #TODO move to Organism class and save the gen_size
                child = dynamic_crossover(prev_gen[i], random.choice(prev_gen))
                cur_gen.append(child)

    best_organism = Organism(best_org_weights)
    best_organism.compile_train(10, X_train, Y_train)

    # test
    y_hat = best_organism.predict(X_test)
    y_hat = y_hat.argmax(axis=1)
    print('Test Accuracy: %.2f' % accuracy_score(Y_test, y_hat))


if __name__ == '__main__':
    main()
