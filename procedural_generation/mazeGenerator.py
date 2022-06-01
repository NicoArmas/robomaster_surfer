import random

import numpy as np
from numpy.random import choice as rc

dic = {0: [0, 1, 2, 3, 4, 5, 6],
       1: [2, 3, 4, 5, 0, 1],
       2: [0, 2],
       3: [0, 1, 2, 3],
       4: [0, 1, 2, 4, 5, 6],
       5: [0, 5, 4, 1],
       6: [0, 6, 4, 2],
       7: []}

#dic = {0: [0, 1, 4, 5],
#       1: [4, 5, 0, 1],
#       2: [0],
#       3: [0, 1],
#       4: [0, 1, 4, 5],
#       5: [0, 5, 4, 1],
#       6: [0, 6, 4],
#       7: []}


def create_row(num):
    a = "{0:b}".format(num)
    b = list(a)
    b = [int(v) for v in b]
    b = [0]*(3-len(b))+b
    return b


def state_from_row(row):
    bin = ""
    for num in row:
        if num == 1:
            bin += "1"
        else:
            bin += "0"
    val = int(bin, 2)
    return val


def generate_lab(initial_state, size, start_from=1, gap=2):
    environment = np.zeros((size, 3), dtype=np.int8)
    coins = [2, 3, 4]
    for i in range(start_from):
        environment[i] = create_row(initial_state)

    count = 0
    choice = 0
    for i in range(start_from-1, len(environment)-1):
        if count == 0:
            choice = random.choice(dic[state_from_row(environment[i])])
        environment[i+1] = create_row(choice)

        for j in range(len(environment[i+1])):
            if environment[i+1][j] == 0:
                if random.random() > 1.9:
                    environment[i+1][j] = rc(coins, 1, [0.8, 0.15, 0.05])

        count += 1
        if count == gap:
            count = 0

    return environment
