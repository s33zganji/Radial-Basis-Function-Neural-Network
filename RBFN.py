import numpy as np
import csv
import math
from matplotlib import pyplot as plt


class RBFN:

    def __init__(self, m, n):

        self.m = m
        self.n = n
        self.w = None
        self.G = None
        self.L = None
        self.error = None
        self.y = None
        self.y_hat = np.array([])
        self.matrix_chrom = np.zeros(shape=(m, n + 1))
        self.end_term = 20


    def input_file(self,t):
        if t==0:
           rand_data = np.random.randint(0, 2000, 1200)
           rand_data.sort()
           with open('reg_data2000.csv') as file:
               reader = csv.reader(file, delimiter=',')
               self.y = np.array([])
               self.x = np.array([])
               list_data = []
               for row in reader:
                   list_data.append(row)
               data = np.array(list_data).astype(np.float)
               list_one = data[:, :-1]
               list_two = data[:, -1]

               self.x = list_one[rand_data]
               self.y = list_two[rand_data]
               max_x = np.max(self.x)
               self.x = self.x / max_x
               self.L = self.x.shape[0]
               self.G = np.zeros(shape=(self.L, self.m))


        if t==1:
           with open('reg_data2000.csv') as file:
                reader = csv.reader(file, delimiter=',')
                self.y = np.array([])
                self.x = np.array([])
                list_data = []
                for row in reader:
                    list_data.append(row)
                data = np.array(list_data).astype(np.float)
                list_one = data[:, :-1]
                list_two = data[:, -1]
                self.x = np.array(list_one)
                self.y = np.array(list_two)
                max_x = np.max(self.x)
                self.x = self.x / max_x
                max_y = np.amax(self.y)
                self.y = self.y / max_y
                self.L = self.x.shape[0]
                self.G = np.zeros(shape=(self.L, self.m))

    def matrix_chromosome(self, individual):
        size = np.array(individual).size
        store = np.zeros(shape=(self.n + 1))
        row = 0
        for i in range(0, size):
            store[i % (self.n + 1)] = individual[i]
            if ((i + 1) % (self.n + 1)) == 0:
                self.matrix_chrom[row] = store
                row += 1


    def G_calculation(self):
        for i in range(0, self.L):
            for j in range(0, self.m):
                v_i = self.matrix_chrom[j, 0: self.n]
                gama = self.matrix_chrom[j, self.n]
                store = (-1) * gama * np.matmul(np.transpose(np.subtract(self.x[i].astype(np.float), v_i)), np.subtract(self.x[i].astype(np.float), v_i))
                self.G[i, j] = math.exp(store)

    def W_calculation(self):
        self.w = np.matmul(np.linalg.pinv(np.matmul(np.transpose(self.G), self.G)), np.matmul(np.transpose(self.G), self.y))

    def y_hat_calculation(self):
        self.y_hat = np.matmul(self.G, self.w)

    def error_calculation(self):
        self.error = np.matmul(np.transpose(np.subtract(self.y_hat, self.y)), np.subtract(self.y_hat, self.y)) / 2
        return self.error

    def run(self, individual):
        individual_numpy = np.array(individual)
        self.matrix_chromosome(individual_numpy)
        self.G_calculation()
        self.W_calculation()
        self.y_hat_calculation()
        return self.error_calculation(),

    def plot(self, t):
      if t==0:
        x_axis = np.array([])
        store = []
        for i in range(0, self.L):
            store.append([i])
            x_axis = np.array(store)

        plt.title("train")
        plt.xlabel("Data arrangement")
        plt.ylabel("y and y'")
        plt.scatter(x_axis, self.y, color='Blue')
        plt.scatter(x_axis, self.y_hat, color="pink")
        plt.show()

      if t==1:
        x_axis = np.array([])
        store = []
        for i in range(0, self.L):
            store.append([i])
            x_axis = np.array(store)

        plt.title("Plot y and y'")
        plt.xlabel("Data arrangement")
        plt.ylabel("test")
        plt.scatter(x_axis, self.y, color='Blue', s=5)
        plt.scatter(x_axis, self.y_hat, color="pink", s=5)
        plt.show()
