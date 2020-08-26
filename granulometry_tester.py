import numpy as np

from cv2 import cv2

from granulo_utils import reshape_img_from_float
from granulo_utils import show_opening_contours
from granulo_utils import plot_histogram
from granulo_utils import threshold_image
from granulo_utils import granulometry_score
from granulo_utils import remove_center



IMG_WIDTH = 768
IMG_HEIGHT = 768
THRESHOLD = 55

# Load OK BSE data
ok_data = np.load("Data/BSE_ok.npy")
# Load also the anomalous BSE data
faulty_data = np.load("Data/BSE_faulty_extended.npy")

# Create structuring element for image opening
struct_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))

okay_scores = []
okay_flagged = 0
faulty_scores = []
faulty_flagged = 0

# For each image, reshape back into 768x768 and stretch, since it's normalized
# Then threshold the image and perform opening, afterwards plot contours
for i in range(0, faulty_data.shape[0]):
    reshaped = reshape_img_from_float(faulty_data[i], IMG_WIDTH, IMG_HEIGHT)
    thresholded = threshold_image(reshaped, THRESHOLD)
    removed = remove_center(thresholded)
    score = granulometry_score(removed, struct_element)
    faulty_scores.append(score)
    if score > 50:
        #show_opening_contours(removed, struct_element, "Faulty")
        faulty_flagged = faulty_flagged + 1
    else:
        cv2.imshow("Undetected Faulty", reshaped)
        cv2.waitKey(0)

for i in range(0, ok_data.shape[0]):
    reshaped = reshape_img_from_float(ok_data[i], IMG_WIDTH, IMG_HEIGHT)
    thresholded = threshold_image(reshaped, THRESHOLD)
    removed = remove_center(thresholded)
    score = granulometry_score(removed, struct_element)
    okay_scores.append(score)
    if score > 50:
        #show_opening_contours(removed, struct_element, "OK")
        okay_flagged = okay_flagged + 1
        cv2.imshow("Falsely flagged OK", reshaped)
        cv2.waitKey(0)

print("Okay")
print(okay_flagged)
print(okay_scores)
print("Faulty")
print(faulty_flagged)
print(faulty_scores)
