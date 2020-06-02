import numpy as np
import matplotlib.pyplot as plt

from Models.BasicAutoencoderEvenDeeper import BasicAutoencoderEvenDeeper
from keras.models import load_model


img_width = 768
img_height = 768

epochs = 50
batch_size = 4
is_data_filtered = "filtered_"


model = BasicAutoencoderEvenDeeper()

# Load the saved model itself
model.model = load_model('Model_Saves/Detailed/filtered_BasicAutoencoderEvenDeeper_e50_b4_detailed')

model.model.summary()

# Load OK_filtered and anomalous data
data = np.load("Data/OK_filtered.npy")
anomalies = np.load("Data/Vadne.npy")

# Reshape to fit the desired input
data = data.reshape(data.shape[0], img_width, img_height, 1)
anomalous_data = anomalies.reshape(anomalies.shape[0], img_width, img_height, 1)

# Normalize the data
train_input = data.astype('float32') / 255.0
anomalous_input = anomalous_data.astype('float32') / 255.0

# Arrays to hold reconstructed images for anoamly detection by mean squared error
reconstructed_ok_array = []
reconstructed_anomalous_array = []

for i in range (0, train_input.shape[0]):
    # Every image needs to be reshaped into 1,768,768,1
    reconstructed_img = model.predict(train_input[i].reshape(1, img_width, img_height, 1))
    # The reconstructed image afterwards needs to be reshaped back into 768 x 768
    reconstructed_img = reconstructed_img.reshape(img_width, img_height)
    # Append the reconstructed image to the reconstructed_ok array
    reconstructed_ok_array.append(reconstructed_img)

# Convert to numpy array
reconstructed_ok_array = np.array(reconstructed_ok_array)

for i in range (0, anomalous_input.shape[0]):
    # Every image needs to be reshaped into 1,768,768,1
    reconstructed_img = model.predict(anomalous_input[i].reshape(1, img_width, img_height, 1))
    # The reconstructed image afterwards needs to be reshaped back into 768 x 768
    reconstructed_img = reconstructed_img.reshape(img_width, img_height)
    # Append the reconstructed image to the reconstructed_ok array
    reconstructed_anomalous_array.append(reconstructed_img)

# Convert to numpy array
reconstructed_anomalous_array = np.array(reconstructed_anomalous_array)

# Array to hold MSE values
reconstructed_ok_errors = []
reconstructed_anomalous_errors = []

# Compute the reconstruction MSE for ok data
for i in range (0, train_input.shape[0]):
    # Reshape into 768x768
    original_image = train_input[i].reshape(img_width, img_height)
    # Add reconstructed image MSE to the array
    reconstructed_ok_errors.append(np.square(np.subtract(original_image, reconstructed_ok_array[i])).mean())

# Same for the anomalous data
for i in range (0, anomalous_input.shape[0]):
    # Reshape into 768x768
    original_image = anomalous_input[i].reshape(img_width, img_height)
    # Add reconstructed image MSE to the array
    reconstructed_anomalous_errors.append(np.square(np.subtract(original_image, reconstructed_anomalous_array[i])).mean())

# Plot the MSEs
x = range (0, len(reconstructed_ok_errors))
z = range (0 + len(reconstructed_ok_errors), len(reconstructed_anomalous_errors) + len(reconstructed_ok_errors))
plt.scatter(x, reconstructed_ok_errors, c='g', s=10, marker='o', edgecolors='black', label='OK')
plt.scatter(z, reconstructed_anomalous_errors, c='r', s=10, marker='o', edgecolors='black', label='Anomalous')
# Horizontal line at 3 times the standard deviation, typical for outlier detection
plt.axhline(y= (3 * np.std(reconstructed_ok_errors)), color='r', linestyle='-')
plt.legend(loc='upper left')
plt.title('model reconstruction error - ' + model.name)
plt.ylabel('Reconstruction Error')
plt.xlabel('Index')
plt.savefig('Graphs/ReconstructionErrors/' + str(is_data_filtered) + model.name + '_e' + str(epochs) + '_b' + str(batch_size) + '_RError.png', bbox_inches = "tight")
plt.close('all')

# Save the error arrays too, so one can see which images were problematic
reconstructed_ok_errors = np.array(reconstructed_ok_errors)
reconstructed_anomalous_errors = np.array(reconstructed_anomalous_errors)

np.save('Reconstructed/Error_Arrays/' + str(is_data_filtered) +  model.name + '_e' + str(epochs) + '_b' + str(batch_size) + '_ROK.npy', reconstructed_ok_errors)
np.save('Reconstructed/Error_Arrays/' + str(is_data_filtered) +  model.name + '_e' + str(epochs) + '_b' + str(batch_size) + '_RAnomalous.npy', reconstructed_anomalous_errors)
