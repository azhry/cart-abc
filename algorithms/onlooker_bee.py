class OnlookerBee:

    def __init__(self, employed_bees):
        self.employed_bees = employed_bees
        self.bees_distribution = np.array([])
        self.food_source_size = len(
            self.employed_bees[0].get_current_food_source())
        self.best_fitness = 0
        self.best_food_source = np.array([])
        self.best_employed_bee = None

    def get_best_food_source(self):
        return self.best_food_source

    def get_best_employed_bee(self):
        return self.best_employed_bee

    def get_best_fitness(self):
        return self.best_fitness

    def evaluates_nectar(self):
        for i in range(len(self.employed_bees)):
            self.employed_bees[i].calculate_fitness()

        self.roulette_wheel()

    def roulette_wheel(self):
        self.bees_distribution = np.array([])
        num_of_bees = len(self.employed_bees)
        total_fitness = 0

        for bee in self.employed_bees:
            fitness = bee.get_current_fitness()
            if fitness > self.best_fitness:
                self.best_food_source = bee.get_current_food_source()
                self.best_fitness = fitness
                self.best_employed_bee = bee
            total_fitness += fitness

        for bee in self.employed_bees:
            freq = int((bee.get_current_fitness() /
                        total_fitness) * num_of_bees)
            for i in range(freq):
                self.bees_distribution = np.append(self.bees_distribution, bee)

        num_of_dist = len(self.bees_distribution)

        for i in range(len(self.employed_bees)):
            self.employed_bees[i] = self.bees_distribution[random.randrange(
                0, num_of_dist - 1)]
