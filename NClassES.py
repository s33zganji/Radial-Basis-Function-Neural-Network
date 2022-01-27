import array
import random
import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from NClass import TwoClass

IND_SIZE = 30  # m*(n+1)
MIN_VALUE = 0
MAX_VALUE = 1
MIN_STRATEGY = 0.01
MAX_STRATEGY = 0.1

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", array.array, typecode="d", fitness=creator.FitnessMin, strategy=None)
creator.create("Strategy", array.array, typecode="d")



def generateES(icls, scls, size, imin, imax, smin, smax):
    ind = icls(random.uniform(imin, imax) for _ in range(size))
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
    return ind


def checkStrategy(minstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children

        return wrappper

    return decorator


two_class = TwoClass(10, 2, 2)
two_class.input(0)
toolbox = base.Toolbox()
toolbox.register("individual", generateES, creator.Individual, creator.Strategy, IND_SIZE, MIN_VALUE, MAX_VALUE,
                 MIN_STRATEGY, MAX_STRATEGY)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxESBlend, alpha=0.1)
toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", two_class.run)
toolbox.decorate("mate", checkStrategy(MIN_STRATEGY))
toolbox.decorate("mutate", checkStrategy(MIN_STRATEGY))


def main():
    random.seed()
    MU, LAMBDA = 10, 100
    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA,
                                              cxpb=0.6, mutpb=0.3, ngen=1, stats=stats, halloffame=hof)
    two_class.plot(0)

    max_ind = 0
    for i in range(0, 10):
        data = pop[i].fitness.values[0]
        if max_ind < data:
            max_ind = data
            individual = pop[i]

    two_class.input(1)

    two_class.run(individual)
    two_class.plot(1)
    return pop, logbook, hof


if __name__ == "__main__":
        main()
