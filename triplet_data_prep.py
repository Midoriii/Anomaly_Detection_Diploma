'''

'''
import numpy as np



def main():
    '''
    Main function that load the BSE and SE OK and Faulty data and calls
    make_triplets() on training partition of the data. Resulting triplets are
    then saved.
    '''
    # Load BSE data - ok and extended faulty, leave ok extra for testing purposes
    bse_ok = np.load("Data/BSE_ok.npy")
    bse_faulty = np.load("Data/BSE_faulty_extended.npy")
    # Load SE data too
    se_ok = np.load("Data/SE_ok.npy")
    se_faulty = np.load("Data/SE_faulty_extended.npy")
    # Use only 80% OK and 75% Faulty data, leave rest for testing
    bse_ok = bse_ok[:int(0.8*len(bse_ok))]
    bse_faulty = bse_faulty[:int(0.75*len(bse_faulty))]
    se_ok = se_ok[:int(0.8*len(se_ok))]
    se_faulty = se_faulty[:int(0.75*len(se_faulty))]
    # Make triplets
    bse_anchor, bse_pos, bse_neg = make_triplets(bse_ok, bse_faulty)
    se_anchor, se_pos, se_neg = make_triplets(se_ok, se_faulty)
    # Save them
    np.save("DataTriplet/BSE_triplet_anchor.npy", bse_anchor)
    np.save("DataTriplet/BSE_triplet_pos.npy", bse_pos)
    np.save("DataTriplet/BSE_triplet_neg.npy", bse_neg)

    np.save("DataTriplet/SE_triplet_anchor.npy", se_anchor)
    np.save("DataTriplet/SE_triplet_pos.npy", se_pos)
    np.save("DataTriplet/SE_triplet_neg.npy", se_neg)


def make_triplets(ok_images, faulty_images):
    '''


    Arguments:
        ok_images: A numpy array of float32 [0,1] values representing OK images.
        faulty_images: A numpy array of float32 [0,1] values representing Faulty images.

    Returns:
        triplets: [anchor, pos, neg] - Three lists of numpy arrays representing images.
    '''


if __name__ == "__main__":
    main()
