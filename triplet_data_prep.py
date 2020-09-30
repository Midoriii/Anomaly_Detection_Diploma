'''

'''
from random import sample

import numpy as np

# Constants
TRIPLETS_NUMBER = 5000


def main():
    '''
    Main function that load the BSE and SE OK and Faulty data and calls
    make_triplets() on training partition of the data. Resulting triplets are
    then saved.
    '''
    # Load BSE data - ok and extended faulty, leave ok extra for testing purposes
    # First try using low dim ones as they worked better for Siamese nets
    bse_ok = np.load("Data/low_dim_BSE_ok.npy")
    bse_faulty = np.load("Data/low_dim_BSE_faulty_extended.npy")
    # Load SE data too
    se_ok = np.load("Data/low_dim_SE_ok.npy")
    se_faulty = np.load("Data/low_dim_SE_faulty_extended.npy")
    # Use only 80% OK and 75% Faulty data, leave rest for testing
    bse_ok = bse_ok[:int(0.8*len(bse_ok))]
    bse_faulty = bse_faulty[:int(0.75*len(bse_faulty))]
    se_ok = se_ok[:int(0.8*len(se_ok))]
    se_faulty = se_faulty[:int(0.75*len(se_faulty))]
    # Make triplets .. three times for good measure
    for i in range(1, 4):
        bse_anchor, bse_pos, bse_neg = make_triplets(bse_ok, bse_faulty)
        se_anchor, se_pos, se_neg = make_triplets(se_ok, se_faulty)
        # Save them
        np.save("DataTriplet/low_dim_BSE_triplet_anchor_" + str(i) + ".npy", bse_anchor)
        np.save("DataTriplet/low_dim_BSE_triplet_pos_" + str(i) + ".npy", bse_pos)
        np.save("DataTriplet/low_dim_BSE_triplet_neg_" + str(i) + ".npy", bse_neg)

        np.save("DataTriplet/low_dim_SE_triplet_anchor_" + str(i) + ".npy", se_anchor)
        np.save("DataTriplet/low_dim_SE_triplet_pos_" + str(i) + ".npy", se_pos)
        np.save("DataTriplet/low_dim_SE_triplet_neg_" + str(i) + ".npy", se_neg)


def make_triplets(ok_images, faulty_images):
    '''


    Arguments:
        ok_images: A numpy array of float32 [0,1] values representing OK images.
        faulty_images: A numpy array of float32 [0,1] values representing Faulty images.

    Returns:
        triplets: [anchor, pos, neg] - Three lists of numpy arrays representing images.
    '''
    # Result lists
    anchor = []
    pos = []
    neg = []
    # Create index lists for image sampling
    ok_idx = list(range(0, ok_images.shape[0]))
    faulty_idx = list(range(0, faulty_images.shape[0]))

    # While number of triplets < desired number
    while len(anchor) < TRIPLETS_NUMBER:
        # Choose between OK or Faulty anchor image by randomly picking a number in range
        # [0, len(ok) + len(faulty)], where if number < len(ok) -> pick OK, else pick Faulty
        ok_or_faulty = sample(list(range(0, ok_images.shape[0] + faulty_images.shape[0])), 1)
        # Pick Anchor by the random index and decide if it's OK or Faulty
        # If the anchor is an OK image ..
        if ok_or_faulty[0] < ok_images.shape[0]:
            # Produce 2 OK samples as anchor and pos, and 1 Faulty as neg
            anchor_pos_samples = sample(ok_idx, 2)
            neg_sample = sample(faulty_idx, 1)
            # Append the data on sampled indexes
            anchor.append(ok_images[anchor_pos_samples[0]])
            pos.append(ok_images[anchor_pos_samples[1]])
            neg.append(faulty_images[neg_sample[0]])
        else:
            # Produce 2 Faulty samples as anchor and pos, and 1 OK as neg
            anchor_pos_samples = sample(faulty_idx, 2)
            neg_sample = sample(ok_idx, 1)
            # Append the data on sampled indexes
            anchor.append(ok_images[anchor_pos_samples[0]])
            pos.append(ok_images[anchor_pos_samples[1]])
            neg.append(faulty_images[neg_sample[0]])

    print(len(anchor))
    print(len(pos))
    print(len(neg))
    # Concat the lists turned to numpy arrays and return them
    return [np.array(anchor), np.array(pos), np.array(neg)]


if __name__ == "__main__":
    main()
