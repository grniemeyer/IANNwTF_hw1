import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# do not need to follow script that closely! ?

''' 2.1 Data '''

digits = load_digits()
#print(digits.data.shape) #(1797, 64) -> already fine?
#print(digits)

# Plot as images
plt.gray()
plt.matshow(digits.images[1])
# plt.show()

# Extract and prepare data
data = digits.data
data = np.float32(data)
data = data/16 # range [0 to 1]
target = digits.target

# Create list of (input, target) tuples
data_target_tuples = list(zip(data, target))
# for i in range(5):
#     print(f"Input: {data_target_tuples[i][0]}, Target: {data_target_tuples[i][1]}")
#print(data_target_tuples, data_target_tuples.shape)

def oneHot(targets):
    unique_t = np.unique(targets)
    one_hot_encoded = []
    for i in targets:
        # for each target entry create one-hot array
        one_hot = np.zeros(len(unique_t))
        one_hot[np.where(unique_t == i)] = 1
        # and append to list
        one_hot_encoded.append(one_hot)

    return one_hot_encoded
target = oneHot(target)

# print(data.shape, type(data)) # ndarray (1797, 64)
# print(type(target)) # list

# convert the target list into an array for convenience reasons
target = np.array(target)


def minibatches_generator(data, targets, minibatchsize):
    '''
    Generator function that shuffles data-target pairs,
    and creates minibatches (subsets of training data) of size minibatchsize.
    Remark: so far the minibatchsize needs to be a factor of len(data) otherwise there are leftover data samples
    
    '''
    # shuffle data-target pairs
    data, targets = shuffle(data, targets)
    # create nr of minibatches with nr of samples in data and roughly minibatchsize (bc of floor division)
    n_minibatches = int(data.shape[0] // minibatchsize)

    # split = int(minibatchsize * data.shape[0])
    # data_batches = np.array(data,64)
    # target_batches = np.array(minibatchsize, 10)

    # infinite loop
    while True:
        # Create minibatches
        for minibatch_idx in range(n_minibatches):
            # for each batch calculate the start and end indices...
            start_idx = minibatch_idx * minibatchsize
            end_idx = (minibatch_idx + 1) * minibatchsize
            # ...to extract the batch from data and target at correct indices
            data_minibatch = data[start_idx:end_idx]
            targets_minibatch = targets[start_idx:end_idx]

            yield data_minibatch, targets_minibatch

generator = minibatches_generator(data, target, 8)
minibatch_data, minibatch_targets = next(generator)

# Print minibatch and shapes of minibatch
# print(minibatch_data, minibatch_targets)
# print(minibatch_data.shape)
# print(minibatch_targets.shape)



''' 2.2 Sigmoid Activation Function '''

class SigmoidActivation:
    def __call__(self, x: np.ndarray):
        self.x: np.ndarray = x

        return 1 / (1 + np.exp(-x))



''' 2.3 Softmax activation function '''

class SoftmaxActivation:
    def __call__(self, x):
        # try different way of checking input parameter
        assert isinstance(x, np.ndarray)
        assert len(x.shape) == 2 # 2D array
        assert x.shape[1] == 10 # 10 units in the second dimension

        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))  # subtracting max for numerical stability?

        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    


''' 2.4 MLP weights '''

class Single_MLP_Layer:
    def __init__(self, activation_function, n_units, input_size):
        """
        - activation_function: SigmoidActivation or SoftmaxActivation
        - n_units: nr of units (perceptrons) in this layer
        - input_size: nr of units in the preceding layer

        """
        self.activation_function = activation_function
        self.num_units = n_units
        self.input_size = input_size

        # initialize weights and bias
        self.weights = np.random.normal(loc=0.0, scale=0.2, size=(input_size, n_units))
        self.bias = np.zeros(n_units)


    def __call__(self, x):
        # forward pass
        self.x = x
        assert isinstance(x, np.ndarray) 
        assert x.shape[1] == self.input_size
        return self.activation_function(self.x @ self.weights + self.bias)



""" 2.5 Putting together the MLP """

class MLP:
    def __init__(self, layer_sizes, activation_functions):
        """
        - layer_sizes: list of ints representing nr of units in each layer
        - activation_functions: list of activation function classes for each layer

        """
        #assert len(layer_sizes) == len(activation_functions)
        #assert len(layer_sizes) > 1

        self.layers = []

        # create MLP layers
        for i in range(1, len(layer_sizes)):
            layer = Single_MLP_Layer(activation_functions[i - 1], layer_sizes[i], layer_sizes[i - 1])
            self.layers.append(layer)


    def __call__(self, x):
        
        """
        - x: array of shape (minibatchsize, 64)
        - output: array of shape (minibatchsize, 10)

        """
        assert isinstance(x, np.ndarray)
        assert x.shape[1] == self.layers[0].input_size, f"Input size must be {self.layers[0].input_size}"

        # forward pass through each layer
        for layer in self.layers:
            x = layer(x)

        return x
        
# Inbetween check a full pass:
minibatch_size = 32
input_size = 64
output_size = 10

# Define layer sizes and activation functions
layer_sizes = [input_size, 128, 64, output_size]
activation_functions = [SigmoidActivation(), SigmoidActivation(), SoftmaxActivation()]

# Create an MLP
mlp = MLP(layer_sizes, activation_functions)

# Assuming minibatch_data has shape (minibatch_size, input_size)
output = mlp(minibatch_data)

# Print the output shape
print("MLP Output Shape:", output.shape)

