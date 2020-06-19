import csv
import numpy as np
import random

training = np.zeros((120, 5), dtype = np.float)
validation = np.zeros((30, 5), dtype = np.float)
input = np.zeros(4, dtype = np.float)
output = np.zeros(3, dtype = np.float)
weight = np.random.normal(0, 0.6, size=(3, 4))
epoch = 0
g = 1
i = 0
j = 0
x = 0

with open('iris.csv', newline='') as csvfile:
    rows = csv.DictReader(csvfile)
    for row in rows:
        num = random.randrange(0, 4)
        if num == 0 and j < 30 or i == 120:
            validation[j][0] = row['sepal.length']
            validation[j][1] = row['sepal.width']
            validation[j][2] = row['petal.length']
            validation[j][3] = row['petal.width']
            j = j + 1
        else:
            training[i][0] = row['sepal.length']
            training[i][1] = row['sepal.width']
            training[i][2] = row['petal.length']
            training[i][3] = row['petal.width']
            i = i + 1

for i in range(3):
   weight[i] = weight[i] / weight[i].sum()

while epoch < 2000:
    num = random.randrange(0, 119)
    for i in range(4):
        input[i] = training[num][i]
    #     x = input.sum()
    # for i in range(4):
    #     input[i] = input[i] / x

    ###### Foward ######

    for j in range(3):
        output[j] = 0
        for i in range(4):
            output[j] += input[i] * weight[j][i]
    # for i in range(3):
    #     print(output[i])
    # print(max(output))
    # print('--------')
    for j in range(3):
        if output[j] == max(output):
            for i in range(4):
                weight[j][i] = weight[j][i] + g * ((input[i]/input.sum()) - weight[j][i])
        else:
            for i in range(4):
                weight[j][i] = weight[j][i]

    epoch = epoch + 1
    # x=0

for x in range(30):
    for i in range(4):
        input[i] = validation[x][i]
    for j in range(3):
        output[j] = 0
        for i in range(4):
            output[j] += input[i] * weight[j][i]
    print(input[0],input[1],input[2],input[3])
    for i in range(3):
        print(i,':',output[i])
    print('---------------------')