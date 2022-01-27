import numpy as np
import csv
from matplotlib import pyplot as plt
import math

class TwoClass:

    def __init__(self, m, n, class_number):

        self.n = n
        self.m = m
        self.w = None
        self.G = None
        self.y = None
        self.L = None
        self.error = None
        self.end_term = 20
        self.class_num = class_number
        self.y_hat = np.array([])
        self.matrix_chrom = np.zeros(shape=(m, n + 1))
        self.max_y_hat = []



    def input(self, t):
      if t==0:
        with open('2clstrain1500.csv') as file:
            reader = csv.reader(file, delimiter=',')
            self.y = np.array([])
            self.x = np.array([])
            list_one = []
            list_two = []
            list_three = []
            max_x = 0
            self.max_y = []
            list_data = []
            for row in reader:
                list_data.append(row)
            data = np.array(list_data).astype(np.float)
            list_one = data[:, :-1]

            self.x = np.array(list_one)
            max_x = np.max(self.x)
            self.x = self.x / max_x
            if self.class_num == 2:
                list_two = data[:, -1]
            elif self.class_num != 2:
                list_three = data[:, -1]

            self.L = self.x.shape[0]


            store = [[0 for x in range(self.class_num)] for y in range(self.L)]
            self.y = np.array(store)

            if self.class_num != 2:
                for i in range(0, self.y.shape[0]):
                    for j in range(0, self.y.shape[1]):

                        temp = list_three[i] - 1

                        if j == temp:
                            self.y[i][j] = 1
                            self.max_y.append(temp + 1)

                        else:
                            self.y[i][j] = 0

            elif self.class_num == 2:
                for i in range(0, self.y.shape[0]):
                    if list_two[i] == [-1]:
                        self.y[i, 0] = 1
                        self.y[i, 1] = 0
                        self.max_y.append(0)
                    else:
                        self.y[i, 0] = 0
                        self.y[i, 1] = 1
                        self.max_y.append(1)

            self.G = np.zeros(shape=(self.L, self.m))

      if t==1:
        with open('2clstrain1500.csv') as file:
            reader = csv.reader(file, delimiter=',')
            self.y = np.array([])
            self.x = np.array([])
            list_one = []
            list_two = []
            list_three = []
            max_x = 0
            self.max_y = []

            list_data = []
            for row in reader:
                list_data.append(row)
            data = np.array(list_data).astype(np.float)
            list_one = data[:, :-1]

            self.x = np.array(list_one)
            max_x = np.max(self.x)
            for i in range(0, self.x.shape[0]):
                self.x[i] = self.x[i] / max_x
            list_two = data[:, -1]
            list_three = data[:, -1]
            self.L = self.x.shape[0]

            store = [[0 for x in range(self.class_num)] for y in range(self.L)]
            self.y = np.array(store)

            if self.class_num != 2:
                for i in range(0, self.y.shape[0]):
                    for j in range(0, self.y.shape[1]):
                        temp = list_three[i] - 1
                        if j == temp:
                            self.y[i][j] = 1
                            self.max_y.append(temp + 1)

                        else:
                            self.y[i][j] = 0

            elif self.class_num == 2:
                for i in range(0, self.y.shape[0]):
                    if list_two[i] == [-1]:
                        self.y[i, 0] = 1
                        self.y[i, 1] = 0
                        self.max_y.append(0)
                    else:
                        self.y[i, 0] = 0
                        self.y[i, 1] = 1
                        self.max_y.append(1)

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
        self.w = np.matmul(np.linalg.pinv(np.matmul(np.transpose(self.G), self.G)),np.matmul(np.transpose(self.G), self.y))

    def y_hat_calculation(self):
        self.max_y_hat = []
        self.y_hat = np.matmul(self.G, self.w)
        if self.class_num == 2:
            self.max_y_hat = np.argmax(self.y_hat, axis=1)
        elif self.class_num != 2:
            self.max_y_hat = np.argmax(self.y_hat, axis=1) + 1

    def error_calculation(self):
        store_y = np.array(self.max_y)
        store_y_hat = np.array(self.max_y_hat)
        subtract= np.subtract(store_y_hat, store_y)
        sum = 0
        accuracy = 0
        for i in range(0, self.L):
            if subtract[i] > 0:
                sum += 1
            elif subtract[i] == 0:
                accuracy = accuracy+1
        self.error = sum / self.L
        print(accuracy/self.L)
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
        store_one = []
        store_two = []
        for i in range(0, self.L):
            store_one.append(self.x[i, 0])
            store_two.append(self.x[i, 1])
        x_axis1 = np.array(store_one)
        x_axis2 = np.array(store_two)

        plt.title("Plot y and y'(test)")
        plt.xlabel("X1")
        plt.ylabel("X2")
        store1 = []
        store2 = []
        store3 = []
        store4 = []
        for i in range(0, self.L):
            if self.max_y_hat[i] == self.max_y[i]:
                store1.append(x_axis1[i])
                store2.append(x_axis2[i])

            else:
                store3.append(x_axis1[i])
                store4.append(x_axis2[i])

        plt.plot(store1, store2, 'ro', color='green')
        plt.plot(store3, store4, 'ro', color='red')

        store5 = []
        store6 = []
        store7 = []
        for i in range(0, self.m):
            store7.append(self.matrix_chrom[i, self.n])
            for j in range(0, self.n):
                if j == 0:
                    store5.append(self.matrix_chrom[i, j])
                else:
                    store6.append(self.matrix_chrom[i, j])

        for i in range(0, self.m):
            circle = plt.Circle((store5[i], store6[i]), store7[i], facecolor='none', edgecolor='black')
            ax = plt.gca()
            ax.add_patch(circle)
            plt.axis('scaled')

        plt.plot(store5, store6, 'ro', color='blue')
        plt.show()

      if t==1:

          store_one = []
          store_two = []
          for i in range(0, self.L):
              store_one.append(self.x[i, 0])
              store_two.append(self.x[i, 1])
          x_axis1 = np.array(store_one)
          x_axis2 = np.array(store_two)

          plt.title("Plot y and y'(train)")
          plt.xlabel("X1")
          plt.ylabel("X2")
          store1 = []
          store2 = []
          store3 = []
          store4 = []
          for i in range(0, self.L):
              if self.max_y_hat[i] == self.max_y[i]:
                  store1.append(x_axis1[i])
                  store2.append(x_axis2[i])

              else:
                  store3.append(x_axis1[i])
                  store4.append(x_axis2[i])

          plt.plot(store1, store2, 'ro', color='green')
          plt.plot(store3, store4, 'ro', color='red')

          store5 = []
          store6 = []
          store7 = []
          for i in range(0, self.m):
              store7.append(self.matrix_chrom[i, self.n])
              for j in range(0, self.n):
                  if j == 0:
                      store5.append(self.matrix_chrom[i, j])
                  else:
                      store6.append(self.matrix_chrom[i, j])

          for i in range(0, self.m):
              circle = plt.Circle((store5[i], store6[i]), store7[i], facecolor='none', edgecolor='black')
              ax = plt.gca()
              ax.add_patch(circle)
              plt.axis('scaled')

          plt.plot(store5, store6, 'ro', color='blue')
          plt.show()
