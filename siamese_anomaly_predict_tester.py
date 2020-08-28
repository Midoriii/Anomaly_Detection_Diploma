import numpy as np

from keras.models import load_model

# Gets predictions for given data with given prototypes
# Faulty and Img_Type denote the correctness of Data and type of Images
def get_predictions(data, data_prototypes, model, faulty="OK", img_type="BSE"):
    IMG_WIDTH = 768
    IMG_HEIGHT = 768

    for i in range(0, data.shape[0]):
        score = 0
        verdict = ""
        # Run through the prototypes
        for j in range(0, data_prototypes.shape[0]):
            score += np.around(model.predict([
                data[i].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1),
                data_prototypes[j].reshape(1, IMG_WIDTH, IMG_HEIGHT, 1)]))

        if score == 5:
            verdict = "OK"
        elif score == 0:
            verdict = "Faulty"
        else:
            verdict = "Don't know"

        print(img_type + " " + faulty + " clonka #" + str(i) + ": " + verdict)


def main():
    # Loading BSE data
    data_ok = np.load("Data/BSE_ok.npy")
    data_faulty = np.load("Data/BSE_faulty.npy")
    data_prototypes = np.load("Data/BSE_prototypes.npy")

    # Loading best BSE Model
    model = load_model("Model_Saves/Detailed/SiameseNetLiteMultipleConv_BSE_e40_b4_detailed", compile=False)

    # First get the predictions for BSE OK and then Faulty images
    get_predictions(data_ok, data_prototypes, model, "OK", "BSE")
    get_predictions(data_faulty, data_prototypes, model, "Faulty", "BSE")

    # Loading SE data
    data_ok = np.load("Data/SE_ok.npy")
    data_faulty = np.load("Data/SE_faulty.npy")
    data_prototypes = np.load("Data/SE_prototypes.npy")
    # Loading best SE model
    model = load_model("Model_Saves/Detailed/BasicSiameseNet_SE_e20_b4_detailed", compile=False)

    # Then the same for SE images
    get_predictions(data_ok, data_prototypes, model, "OK", "SE")
    get_predictions(data_faulty, data_prototypes, model, "Faulty", "SE")


if __name__ == "__main__":
    main()
