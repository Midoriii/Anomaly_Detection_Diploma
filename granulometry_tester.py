import numpy as np

from cv2 import cv2

from granulo_utils import reshape_img_from_float
from granulo_utils import show_opening
from granulo_utils import plot_histogram
from granulo_utils import threshold_image
from granulo_utils import adaptive_threshold_image
from granulo_utils import granulometry_score
from granulo_utils import remove_center



IMG_WIDTH = 768
IMG_HEIGHT = 768
# For binary thresholding
THRESHOLD = 55
# Score which indicated anomaly if val > score
SCORE_THRESHOLD = 50
# For adaptive thtesholding
GROUP_SIZE = 151
SUBTRACT_CONST = 60

# Load OK BSE data
ok_data = np.load("Data/BSE_ok.npy")
# Load also the anomalous BSE data
faulty_data = np.load("Data/BSE_faulty_extended.npy")

# Create structuring element for image opening
struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

okay_scores = []
# To count how many okay images were incorrectly flagged as faulty
okay_flagged = 0
faulty_scores = []
# To count how many faulty images were correctly flagged as faulty
# Should probably be how many weren't flagged but whatever
faulty_flagged = 0

# For each image, reshape back into 768x768 and stretch, since they're normalized
# Then threshold the image and perform opening
# Afterwards plot contours (currently commented out)
for i in range(0, faulty_data.shape[0]):
    reshaped = reshape_img_from_float(faulty_data[i], IMG_WIDTH, IMG_HEIGHT)
    thresholded = threshold_image(reshaped, THRESHOLD)
    #thresholded = adaptive_threshold_image(reshaped, GROUP_SIZE, SUBTRACT_CONST)
    removed = remove_center(thresholded)
    score = granulometry_score(removed, struct_element)
    #show_opening(removed, struct_element, "Faulty")
    faulty_scores.append(score)
    if score > SCORE_THRESHOLD:
        #show_opening(removed, struct_element, "Faulty")
        faulty_flagged = faulty_flagged + 1
    else:
        cv2.imshow("Undetected Faulty", reshaped)
        cv2.waitKey(0)

for i in range(0, ok_data.shape[0]):
    reshaped = reshape_img_from_float(ok_data[i], IMG_WIDTH, IMG_HEIGHT)
    thresholded = threshold_image(reshaped, THRESHOLD)
    #thresholded = adaptive_threshold_image(reshaped, GROUP_SIZE, SUBTRACT_CONST)
    removed = remove_center(thresholded)
    score = granulometry_score(removed, struct_element)
    okay_scores.append(score)
    if score > SCORE_THRESHOLD:
        #show_opening(removed, struct_element, "OK")
        okay_flagged = okay_flagged + 1
        cv2.imshow("Falsely flagged OK", reshaped)
        cv2.waitKey(0)

print("Okay")
print(okay_flagged)
print(okay_scores)
print("Faulty")
print(faulty_flagged)
print(faulty_scores)
