class EmployedBee:

    def __init__(self, dataset, labels, test_data, test_labels, starter_food_source, modification_rate, k=3):
        self.current_limit = 0
        self.MAX_LIMIT = 3
        self.dataset = dataset
        self.labels = labels
        self.test_data = test_data
        self.test_labels = test_labels
        self.features = self.dataset.columns
        self.current_food_source = starter_food_source
        self.modification_rate = modification_rate
        self.k = k

        knn = KNeighborsClassifier(n_neighbors=k)
        selected_features = [f for i, f in enumerate(
            self.features) if self.current_food_source[i] == 1]
        # knn.fit(self.dataset[selected_features].to_numpy(), self.labels.to_numpy())
        # self.current_fitness = knn.score(self.test_data[selected_features].to_numpy(), self.test_labels.to_numpy())
        # self.y_pred = knn.predict(self.test_data[selected_features].to_numpy())

        self.y_pred = get_mknn_predicted(self.dataset[selected_features].to_numpy(), self.labels.to_numpy(
        ), self.test_data[selected_features].to_numpy(), self.test_labels.to_numpy(), k)
        self.current_fitness = accuracy_score(
            self.test_labels.to_numpy(), self.y_pred)

    def calculate_fitness(self):
        knn = KNeighborsClassifier(n_neighbors=self.k)
        neighbor = [(1 if random.uniform(0, 1) < self.modification_rate else bit)
                    for bit in self.current_food_source]
        selected_features = [f for i, f in enumerate(
            self.features) if neighbor[i] == 1]
        # knn.fit(self.dataset[selected_features].to_numpy(), self.labels.to_numpy())

        predicted = get_mknn_predicted(self.dataset[selected_features].to_numpy(), self.labels.to_numpy(
        ), self.test_data[selected_features].to_numpy(), self.test_labels.to_numpy(), self.k)
        neighbor_fitness = accuracy_score(
            self.test_labels.to_numpy(), predicted)
        # neighbor_fitness = knn.score(self.test_data[selected_features].to_numpy(), self.test_labels.to_numpy())
        # self.y_pred = knn.predict(self.test_data[selected_features].to_numpy())

        self.y_pred = predicted
        if neighbor_fitness > self.current_fitness:
            self.current_limit = 0
            self.current_food_source = neighbor
            self.current_fitness = neighbor_fitness
        else:
            self.current_limit += 1
            if self.current_limit != self.MAX_LIMIT:
                self.calculate_fitness()
            else:
                self.current_limit = 0
                self.current_food_source = np.array(
                    [random.choice((0, 1)) for _ in range(len(self.current_food_source))])

    def get_y_pred(self):
        return self.y_pred

    def get_current_fitness(self):
        return self.current_fitness

    def get_current_food_source(self):
        return self.current_food_source

    def generate_new_food_source(self):
        self.current_food_source = np.array(
            [random.choice((0, 1)) for _ in range(len(self.current_food_source))])
