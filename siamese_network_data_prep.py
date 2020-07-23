import numpy as np

''' Run this only if you have ~30GiB of RAM available, since the
    data is by all means huge (tens of thousands of 768x768 images)
'''


img_width = 768
img_height = 768

# Load the already saved OK data
part1 = np.load("Data/OK_1.npy")
part2 = np.load("Data/OK_2.npy")
data = np.concatenate((part1, part2))

print(data.shape)

# Load the anomalies too
anomalies = np.load("Data/Vadne.npy")
print(anomalies.shape)

# Reshape to fit the desired input
data = data.reshape(data.shape[0], img_width, img_height, 1)
anomalous_data = anomalies.reshape(anomalies.shape[0], img_width, img_height, 1)

# Create pairs for the siamese net input
# Initial effort is to create each with each pairing
pairs_left = []
pairs_right = []
pairs_labels = []

# Create pairs from OK data
# shape - 1 because the last one shouldn't need to be paired again
for left in range(data.shape[0]-1):
    # left + 1 to not make duplicate pairs
    for right in range(left+1, data.shape[0]):
        pairs_left.append(data[left])
        pairs_right.append(data[right])
        pairs_labels.append(1)

# Create pairs from OK and anomalous data
for left in range(data.shape[0]):
    for right in range(anomalous_data.shape[0]):
        pairs_left.append(data[left])
        pairs_right.append(anomalous_data[right])
        pairs_labels.append(0)

# Recreate as numpy arrays
pairs_left = np.array(pairs_left)
pairs_right = np.array(pairs_right)
pairs_labels = np.array(pairs_labels)

# Shuffle the data before saving
indices = np.arange(pairs_left.shape[0])
np.random.shuffle(indices)

pairs_left = pairs_left[indices]
pairs_right = pairs_right[indices]
pairs_labels = pairs_labels[indices]

print(pairs_labels.shape)
print(pairs_left.shape)
print(pairs_right.shape)

# Normalize the data
pairs_left = pairs_left.astype('float32') / 255.0
pairs_right = pairs_right.astype('float32') / 255.0

# Save the data for future usage
np.save('DataHuge/Pairs_Left.npy', pairs_left)
np.save('DataHuge/Pairs_Right.npy', pairs_right)
np.save('DataHuge/Pairs_Labels.npy', pairs_labels)
