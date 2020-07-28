import glob
import numpy as np
import cv2


def crop_reshape(images):
    images_list = []
    img_width = 768
    img_height = 768

    for image in images:
        img = cv2.imread(image)
        # Resize into 768*768 if bigger
        resized = cv2.resize(img, (768, 840))
        # Crop the bottom info
        cropped_img = resized[0:768, 0:768]
        # Make it actual grayscale
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        # Add the new grayscale image to the list
        images_list.append(gray)
    # As numpy array
    images_list = np.array(images_list)
    # Return the list for further operations
    return images_list


def make_pairs(ok_images, faulty_images):
    # Create pairs for the siamese net input
    # Initial effort is to create each with each pairing
    pairs_left = []
    pairs_right = []
    pairs_labels = []
    img_width = 768
    img_height = 768

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
    pairs_left = pairs_left.astype('float32') / 255.0
    pairs_right = pairs_right.astype('float32') / 255.0

    # Reshape into desirable shape
    pairs_left = pairs_left.reshape(pairs_left.shape[0], img_width, img_height, 1)
    pairs_right = pairs_right.reshape(pairs_right.shape[0], img_width, img_height, 1)

    return pairs_left, pairs_right, pairs_labels


def main():
    ok_images_se = glob.glob('Clonky-ok/*_3*')
    ok_images_bse = glob.glob('Clonky-ok/*_4*')

    # Apparently glob is incapable of such matching, that 'b' could be excluded
    faulty_images_se = glob.glob('Clonky-vadne/[0-9].*')
    faulty_images_se += glob.glob('Clonky-vadne/[0-9][0-9].*')
    faulty_images_bse = glob.glob('Clonky-vadne/[0-9]*b.*')

    ok_images_se_list = crop_reshape(ok_images_se)
    ok_images_bse_list = crop_reshape(ok_images_bse)
    faulty_images_se_list = crop_reshape(faulty_images_se)
    faulty_images_bse_list = crop_reshape(faulty_images_bse)

    print(ok_images_se_list.shape)
    print(ok_images_bse_list.shape)
    print(faulty_images_se_list.shape)
    print(faulty_images_bse_list.shape)

    # Save even the sorted data into SE and BSE types
    np.save("Data/SE_ok.npy", ok_images_se_list)
    np.save("Data/BSE_ok.npy", ok_images_bse_list)
    np.save("Data/SE_faulty.npy", faulty_images_se_list)
    np.save("Data/BSE_faulty.npy", faulty_images_bse_list)

    # Leave some of the images as testing data
    # 75% instead of 80% of faulty data is taken to represent all the error types
    ok_images_se_list = ok_images_se_list[:int(0.8*len(ok_images_se_list))]
    ok_images_bse_list = ok_images_bse_list[:int(0.8*len(ok_images_bse_list))]
    faulty_images_se_list = faulty_images_se_list[:int(0.75*len(faulty_images_se_list))]
    faulty_images_bse_list = faulty_images_bse_list[:int(0.75*len(faulty_images_bse_list))]

    print(ok_images_se_list.shape)
    print(ok_images_bse_list.shape)
    print(faulty_images_se_list.shape)
    print(faulty_images_bse_list.shape)

    se_pairs_left, se_pairs_right, se_pairs_labels = make_pairs(ok_images_se_list, faulty_images_se_list)
    bse_pairs_left, bse_pairs_right, bse_pairs_labels = make_pairs(ok_images_bse_list, faulty_images_bse_list)

    np.save("DataHuge/SE_pairs_left.npy", se_pairs_left)
    np.save("DataHuge/SE_pairs_right.npy", se_pairs_right)
    np.save("DataHuge/SE_pairs_labels.npy", se_pairs_labels)

    np.save("DataHuge/BSE_pairs_left.npy", bse_pairs_left)
    np.save("DataHuge/BSE_pairs_right.npy", bse_pairs_right)
    np.save("DataHuge/BSE_pairs_labels.npy", bse_pairs_labels)

    return


if __name__ == "__main__":
    main()
