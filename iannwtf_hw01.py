import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

# do not need to follow script that closely! ?


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
    ''' generator function that shuffles data-target pairs
    and creates minibatches (subsets of training data) of size minibatchsize '''
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
print(minibatch_data, minibatch_targets)
print(minibatch_data.shape)
print(minibatch_targets.shape)

