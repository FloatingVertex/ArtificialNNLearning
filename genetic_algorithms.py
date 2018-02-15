import numpy as np
import NNImplementation as nn
import problems

def train_weights(weights, problem,funct,learning_rate=0.01):
    """
    train a list of weights one step given the problem and eval function, return new weights
    :param weights:     the weigts to use as a starting point
    :param problem:     the problem to solve, output to this will be maximized
    :param funct:       the eval function to use on the weights
    :return:            the new set of weights
    """
    # randomly generate a list of new candidates
    new_candidates = [weights]
    for _ in range(30):
        new_candidates.append(nn.randomly_adjust_weights_sum(weights,learning_rate))
    for _ in range(30):
        new_candidates.append(nn.randomly_adjust_weights_mult(weights,learning_rate))
    # generate scores for each candidate
    scores = []
    for candidate in new_candidates:
        candidate_function = nn.run_with_weights(funct,candidate)
        scores.append(problem(candidate_function))

    # get tuple of (max_score,max_score_weights)
    scores_candidates = list(zip(scores,new_candidates))
    max_score_set = max(scores_candidates,key=lambda item:item[0])
    # return only the weights
    return max_score_set[1]