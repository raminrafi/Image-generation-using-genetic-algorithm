import random
import numpy as np
from PIL import Image
from deap import base, creator, tools, algorithms

POPULATION_SIZE = 100

class GeneticAlgorithm:
    def __init__(self, image_name):
        self.target_image = Image.open(image_name)
        self.pix = self.target_image.load()

        self.current_pixel = [0, 0]

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)  # CREATING POPULATION

        self.toolbox = base.Toolbox()
        self.toolbox.register("attr_rgb", self.__create_gnome)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_rgb, n=3)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Register toolbox functions
        self.toolbox.register("evaluate", self.__cal_fitness)  # CALCULATING FITNESS
        self.toolbox.register("mate", tools.cxTwoPoint)  # TWOPOINT CROSSOVER
        self.toolbox.register("mutate", self.__mutation, indpb=1 / 3)  # MUTATION
        self.toolbox.register("select", tools.selTournament, tournsize=3)  # SELECTION

    # Create value
    def __create_gnome(self):
        return random.randint(0, 255)

    # Score individual fitness
    def __cal_fitness(self, individual):
        current_x, current_y = self.current_pixel

        tr, tg, tb = self.pix[current_x, current_y]
        ir, ig, ib = tuple(individual)

        fitness = abs(tr - ir) + abs(tg - ig) + abs(tb - ib)
        return fitness,

    # Mutate genes
    def __mutation(self, individual, indpb):
        for i in range(len(individual)):
            individual[i] = self.__create_gnome() if random.random() <= indpb else individual[i]
        return individual,

    def main(self):
        w, h = self.target_image.size
        rpixels = []

        count = 1
        for y in range(h):
            for x in range(w):
                print("Evolving pixel %s of %s" % (count, w * h))
                count += 1

                if (count == 2200):
                    showImage(rpixels)
                elif (count == 4400):
                    showImage(rpixels)
                elif (count == 6600):
                    showImage(rpixels)

                self.current_pixel = [x, y]

                populate = self.toolbox.population(n=10)
                hallofFame = tools.HallOfFame(maxsize=1)
                stats = tools.Statistics(lambda ind: ind.fitness.values)
                stats.register("min", np.min)

                algorithms.eaSimple(population=populate, toolbox=self.toolbox, cxpb=0.5, mutpb=0.2, ngen=50,
                                    stats=stats, halloffame=hallofFame, verbose=False)

                rpixels.append(hallofFame[0])
        return rpixels


def showImage(output):
    fpixels = []
    for pix in output:
        fpixels.append(tuple(pix))

    res = Image.new(mode=OBJ.target_image.mode, size=OBJ.target_image.size)
    res.putdata(data=fpixels)
    res.save("res.png")
    res.show()


OBJ = GeneticAlgorithm(image_name="imageB.jpg")
opixels = OBJ.main()
showImage(opixels)