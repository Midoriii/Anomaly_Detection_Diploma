'''
The purpose of this script is to try granulometry on BSE type images, both OK
and faulty ones.

Granulometry could prove to be a simple and useful addition to Autoencoders,
which struggle with defects that manifest as black dots on the image.
Structuring element of good enough shape and size could filter out small, acceptable
dots, while leaving the ones large enough to be marked as defects. Those are then
summed up (the essence of granulometry) and if too many pixels remain, the image
can be labeled as faulty.

Normalized, float32 numpy arrays of images are loaded and afterwards stretched
back into 768x768 and denormalized back to [0,255]. The grayscale images are then
thresholded using either binary or gaussian adaptive thresholding, since we want
only a binary image for image opening and granulometry. On BSE images, defects
manifest as black shapes, so the goal of thresholding is to isolate those.

The central part of thresholded image is then cut out, because it usually
contains the central hole of the shutter, which messes up the granulometry. Not
too many, if any, defects show in this small area, so it's rather safe to do so.

Finally, binary image opening is performed with the defined structuring element.
Pixels remaining are then counted and the sum functions as score of sorts. It's
important to find out good enough threshold for this score, and it heavily depends
on the thresholding performed and its parameters, as well as on the structuring element
used and its size.

Current version shows the original image of both false positives and false negatives.

Best performing params found so far are set as constants in this script. Unfortunately,
it seems that the line between acceptable and faulty defects is too thin to be
captured perfectly and even granulometry still leaves some false negatives. False
positives are a lesser issue, but can stem from exceptionally low quality input image.

@Author: Stepan Benes
'''
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
