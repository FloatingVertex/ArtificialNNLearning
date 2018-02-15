import numpy as np
import random

def output_1(funct):
    """
    Simple problem to test if learning algorithm is working
    :param funct:   a function that can take a list of 2 inputs and returns a list of 1 output
    :return:        the score, max possible is 10
    """
    scores = []
    for _ in range(10):
        input = [[random.uniform(0,4)],[1]]
        output = np.mean(funct(input))
        scores.append(10-abs(output-1))
    return sum(scores)/len(scores)
