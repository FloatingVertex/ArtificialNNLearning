import numpy as np

def relu(inputs):
    # relu function, the implementation is faster than np.max()
    return inputs * (inputs > 0)

def apply_layer(inputs,weights):
    return np.matmul(weights,inputs)

def random_weights(input_shape,output_shape):
    return np.random.uniform(0,1,output_shape+input_shape)

def create_weights(inputs,layers):
    layers = [inputs] + layers
    weights = [random_weights(first,second) for first,second in zip(layers[:-1],layers[1:])]
    return weights

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

weights = create_weights((4,),[(10,),(4,),(2,)])
print(run_with_relu(np.asarray([[.2],[.1],[.1],[.3]]),weights))
weights = randomly_adjust_weights_list_sum(weights)
print(run_with_relu(np.asarray([[.2],[.1],[.1],[.3]]),weights))