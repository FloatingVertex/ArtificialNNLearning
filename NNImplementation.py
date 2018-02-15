import numpy as np

def relu(inputs):
    # relu function, the implementation is faster than np.max()
    return inputs * (inputs > 0)

def apply_layer(inputs,weights):
    return np.matmul(weights,inputs)

def random_weights(input_shape,output_shape):
    return np.random.uniform(0,1,output_shape+input_shape)

def combine_weights(first_set,second_set,first_portion):
    """
    Combines two sets of weights randomly to generate a new unique set of weights
    :param first_set:       first set of weights
    :param second_set:      second set of weights
    :param first_portion:   chance of any particular value being selected from the first set range 0-1
    :return:                newly generated weights with some elments from each input set
    """
    if first_set.shape != second_set.shape:
        raise ValueError("First and Second set of weights must have same shape")
    random_weights = np.random.uniform(0,1,second_set.shape)
    new_weights = (random_weights < first_portion)*first_set + (random_weights >= first_portion)*second_set
    return new_weights

def create_weights(inputs,layers):
    layers = [inputs] + layers
    weights = [random_weights(first,second) for first,second in zip(layers[:-1],layers[1:])]
    return np.asarray(weights)

def randomly_adjust_weights_mult(weights,range = 0.01):
    #multiply pair wise, * does dot product
    return np.multiply(weights,np.random.uniform(1-range,1+range, weights.shape))

def randomly_adjust_weights_sum(weights,range = 0.01):
    return weights + np.random.uniform(-range, range, weights.shape)

def randomly_adjust_weights_list_sum(weights,range = 0.01):
    return [randomly_adjust_weights_sum(layer_weights,range) for layer_weights in weights]

def randomly_adjust_weights_list_mult(weights,range = 0.01):
    return [randomly_adjust_weights_mult(layer_weights,range) for layer_weights in weights]

def run_with_relu(inputs,weights):
    current_values = inputs
    # pass inputs through the neural netowrk using given weights
    for layer_weight in weights:
        current_values = apply_layer(current_values,layer_weight)
        # relu used as activation function
        current_values = relu(current_values)
    outputs = current_values
    return outputs

def run_with_weights(funct,weights):
    """
    create a new function that runs the given function with given weights in form funct(<inputs passed in>, weights)
    :param funct:       the function (i.e. run_with_relu)
    :param weights:     the weights, the weights to alwase use when running the above function
    :return:            a function that can be used like so: returned_function(<inputs>) and it will run funct(<inputs>,weights)
    """
    def new_function(input):
        input = np.asarray(input)
        return funct(input,weights)
    return new_function