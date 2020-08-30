import sys
import glob
import numpy as np

from reshape_util import crop_reshape
from reshape_util import reshape_normalize


def make_pairs(ok_images, faulty_images):
    # Create pairs for the siamese net input
    # Initial effort is to create 'each with each' pairing
    pairs_left = []
    pairs_right = []
    pairs_labels = []

    IMG_WIDTH = 768
    IMG_HEIGHT = 768

    # Create pairs from OK images
    # shape - 1 because the last one shouldn't need to be paired again
    for left in range(ok_images.shape[0]-1):
        # left + 1 to not make duplicate pairs
        for right in range(left+1, ok_images.shape[0]):
            pairs_left.append(ok_images[left])
            pairs_right.append(ok_images[right])
            pairs_labels.append(1)

    # Create pairs from OK and anomalous imaes
    for left in range(ok_images.shape[0]):
        for right in range(faulty_images.shape[0]):
            pairs_left.append(ok_images[left])
            pairs_right.append(faulty_images[right])
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
    pairs_left = reshape_normalize(pairs_left, IMG_WIDTH, IMG_HEIGHT)
    pairs_right = reshape_normalize(pairs_right, IMG_WIDTH, IMG_HEIGHT)

    return pairs_left, pairs_right, pairs_labels


def main():
    IMG_WIDTH = 768
    IMG_HEIGHT = 768

    ok_images_se = glob.glob('Clonky-ok/*_3*')
    ok_images_bse = glob.glob('Clonky-ok/*_4*')

    extended = ""

    # A cheap way to make a distinction between making pairs with all the faulty
    # images (which means even those that contain plugged center) - if any
    # argument is given, all the faulty images are used.
    if len(sys.argv) > 1:
        # Apparently glob is incapable of such matching, that 'b' could be excluded
        # So I have to use this gimmick with 'numbers followed by dot'
        faulty_images_se = glob.glob('Clonky-vadne-full/[0-9].*')
        faulty_images_se += glob.glob('Clonky-vadne-full/[0-9][0-9].*')
        faulty_images_bse = glob.glob('Clonky-vadne-full/[0-9]*b.*')
        extended = "_extended"
    else:
        faulty_images_se = glob.glob('Clonky-vadne/[0-9].*')
        faulty_images_se += glob.glob('Clonky-vadne/[0-9][0-9].*')
        faulty_images_bse = glob.glob('Clonky-vadne/[0-9]*b.*')

    # Remove the info bar from the images and reshape them into 768x768
    ok_images_se_list = crop_reshape(ok_images_se)
    ok_images_bse_list = crop_reshape(ok_images_bse)
    faulty_images_se_list = crop_reshape(faulty_images_se)
    faulty_images_bse_list = crop_reshape(faulty_images_bse)

    print(ok_images_se_list.shape)
    print(ok_images_bse_list.shape)
    print(faulty_images_se_list.shape)
    print(faulty_images_bse_list.shape)

    # Save all of the data, divided by the image type
    np.save("Data/SE_ok.npy", reshape_normalize(ok_images_se_list, IMG_WIDTH, IMG_HEIGHT))
    np.save("Data/BSE_ok.npy", reshape_normalize(ok_images_bse_list, IMG_WIDTH, IMG_HEIGHT))
    np.save("Data/SE_faulty" + extended + ".npy", reshape_normalize(faulty_images_se_list, IMG_WIDTH, IMG_HEIGHT))
    np.save("Data/BSE_faulty" + extended + ".npy", reshape_normalize(faulty_images_bse_list, IMG_WIDTH, IMG_HEIGHT))

    # Leave some of the images as testing data.
    # 75% instead of 80% of faulty data is taken to leave enough images
    # to represent all the error types in the testing portion.
    ok_images_se_list = ok_images_se_list[:int(0.8*len(ok_images_se_list))]
    ok_images_bse_list = ok_images_bse_list[:int(0.8*len(ok_images_bse_list))]
    faulty_images_se_list = faulty_images_se_list[:int(0.75*len(faulty_images_se_list))]
    faulty_images_bse_list = faulty_images_bse_list[:int(0.75*len(faulty_images_bse_list))]

    print(ok_images_se_list.shape)
    print(ok_images_bse_list.shape)
    print(faulty_images_se_list.shape)
    print(faulty_images_bse_list.shape)

    # Make input pairs out of the training data portion
    se_pairs_left, se_pairs_right, se_pairs_labels = make_pairs(ok_images_se_list, faulty_images_se_list)
    bse_pairs_left, bse_pairs_right, bse_pairs_labels = make_pairs(ok_images_bse_list, faulty_images_bse_list)

    # And, of course, save them
    np.save("DataHuge/SE_pairs_left" + extended + ".npy", se_pairs_left)
    np.save("DataHuge/SE_pairs_right" + extended + ".npy", se_pairs_right)
    np.save("DataHuge/SE_pairs_labels" + extended + ".npy", se_pairs_labels)

    np.save("DataHuge/BSE_pairs_left" + extended + ".npy", bse_pairs_left)
    np.save("DataHuge/BSE_pairs_right" + extended + ".npy", bse_pairs_right)
    np.save("DataHuge/BSE_pairs_labels" + extended + ".npy", bse_pairs_labels)


if __name__ == "__main__":
    main()
