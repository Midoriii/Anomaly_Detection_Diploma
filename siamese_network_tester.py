import numpy as np
import cv2
import sys, getopt
import matplotlib.pyplot as plt

from PIL import Image
from Models.BasicSiameseNet import BasicSiameseNet


img_width = 768
img_height = 768
input_shape = (img_width, img_height, 1)
# Default params
epochs = 20
batch_size = 4
image_type = 'SE'
desired_model = "BasicSiameseNet"

# Get full command-line arguments
full_cmd_arguments = sys.argv
# Keep all but the first
argument_list = full_cmd_arguments[1:]
# Getopt options
short_options = "e:m:t:"
long_options = ["epochs=", "model=", "type="]
# Get the arguments and their respective values
arguments, values = getopt.getopt(argument_list, short_options, long_options)

# Evaluate given options
for current_argument, current_value in arguments:
    if current_argument in ("-e", "--epochs"):
        epochs = int(current_value)
    elif current_argument in ("-m", "--model"):
        desired_model = current_value
    elif current_argument in ("-t", "--type"):
        image_type = current_value


# Loading data and labels - BSE or SE image origin as chosen in args
if image_type == 'BSE':
    left_data = np.load('DataHuge/BSE_pairs_left.npy')
    right_data = np.load('DataHuge/BSE_pairs_right.npy')
    labels = np.load('DataHuge/BSE_pairs_labels.npy')
elif image_type == 'SE':
    left_data = np.load('DataHuge/SE_pairs_left.npy')
    right_data = np.load('DataHuge/SE_pairs_right.npy')
    labels = np.load('DataHuge/SE_pairs_labels.npy')
else:
    print("Wrong Image Type specified!")
    sys.exit()
# Normalization not needed, data is already normalized


# Choose desired model
if desired_model = "BasicSiameseNet":
    model = BasicSiameseNet()


# Create and compile the model
model.create_net(input_shape)
model.compile_net()
#Train it
model.train_net(left_data, right_data, labels, epochs=epochs, batch_size=batch_size)

plt.plot(model.history.history['loss'])
plt.title('model loss - ' + model.name)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('Graphs/Losses/' + model.name + '_' + str(image_type) + '_e' + str(epochs) + '_b' + str(batch_size) + '_loss.png', bbox_inches = "tight")
plt.close('all')

plt.plot(model.history.history['accuracy'])
plt.title('model accuracy - ' + model.name)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig('Graphs/Accuracies/' + model.name + '_' + str(image_type) + '_e' + str(epochs) + '_b' + str(batch_size) + '_acc.png', bbox_inches = "tight")
plt.close('all')

prediction = model.predict(right_data[24].reshape(1, img_width, img_height, 1))
print(prediction)
