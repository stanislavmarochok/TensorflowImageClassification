import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import sklearn
import sklearn.preprocessing as ppc
import random
import pickle
import json
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, LeakyReLU
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.python.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard

DATADIR = "d:\Study\Ing\\1 semester\student\I-SUNS\Zadanie 5\\files\data\images"
IMG_SIZE = 50
CONV_NETWORK_MODEL_SAVED_DIR = "conv_network_model_saved"
CONV_NETWORK_MODEL_NAME = "conv_network_model.h5"
CONV_NETWORK_FIT_HISTORY = "conv_network_git_history"

# change this variable to a category you want to classify images
# possible values are "gender", "masterCategory", "usage", "season"
CLASSIFICATION_CATEGORY = "masterCategory"


# *****************************************************************************
# This function prints unique values from *column of *df (DataFrame)
# *****************************************************************************
def print_unique_values_for_column(df, column):
    unique_values = df[column].unique().tolist()
    print("{: >15} {: 4} ".format(column, len(unique_values)), unique_values)


# *****************************************************************************
# This function is used for printing unique values from DataFrame
# *****************************************************************************
def print_unique_values(df):
    print_unique_values_for_column(df, 'gender')
    print_unique_values_for_column(df, 'masterCategory')
    print_unique_values_for_column(df, 'season')
    print_unique_values_for_column(df, 'usage')


# ******************************************************************************
# This function saves datasets X and y to files X.pickle and y.pickle
# With the help of library pickle
# ******************************************************************************
def save_dataset(X, y):
    pickle_out = open("X.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("y.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()


# ******************************************************************************
# This function saves datasets X and y to files X.pickle and y.pickle
# With the help of library pickle
# ******************************************************************************
def save_model(model, history):
    # create sub  folder if not exists
    if not os.path.exists(CONV_NETWORK_MODEL_SAVED_DIR):
        os.makedirs(CONV_NETWORK_MODEL_SAVED_DIR)

    # save model
    model.save(os.path.join(CONV_NETWORK_MODEL_SAVED_DIR, CONV_NETWORK_MODEL_NAME))

    # save fit history as json dictionary
    with open(os.path.join(CONV_NETWORK_MODEL_SAVED_DIR, CONV_NETWORK_FIT_HISTORY), mode='w') as f:
        json.dump(history, f)


# ******************************************************************************
# This function prints a plot with a history of Convolution Network
# Safely copy-pasted from
# https://www.datacamp.com/community/tutorials/convolutional-neural-networks-python
# ******************************************************************************
def show_plot(history):
    accuracy = history['accuracy']
    val_accuracy = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
    epochs = range(len(accuracy))

    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


# ******************************************************************************
# This function visualizes filters of a model of a Convolution Network
# Carefully copy-pasted from
# https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
# ******************************************************************************
def visualize_filters(model):
    filters, biases = model.layers[4].get_weights()

    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)
    n_filters, ix = 10, 1

    for i in range(n_filters):
        f = filters[:, :, :, i]
        for j in range(3):
            ax = plt.subplot(n_filters, 3, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(f[:, :, j], cmap='gray')
            ix += 1
    plt.show()


# ******************************************************************************
# This function is used to test predictions of a trained neural network
# on my own image (not from dataset)
# ******************************************************************************
def test_conv_network_on_image(image_path):
    if not os.path.exists(image_path):
        print("Image", image_path, "not exists")
        return

    feature = []
    try:
        img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
        plt.imshow(img_array, cmap="gray")
        plt.show()
        feature.append(img_array)
    except Exception as e:
        print("Some error occurred while preparing an image")
        return

    feature = np.array(feature).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    # normalizing
    feature = np.array(feature / 255.0)

    # image is prepared, now lets test our network
    model, history = get_convolutional_network_model()
    predictions = model.predict(feature)
    predicted_labels = np.argmax(predictions, axis=1)

    # get a label encoder to get a text value of predicted value
    df = get_dataframe(False)
    le = ppc.LabelEncoder()
    le.fit(df[CLASSIFICATION_CATEGORY])
    # getting all unique values for our special classification category from the dataset
    # converting them to text, numbers, and one-hot representation
    unique_values_text = np.unique(df[CLASSIFICATION_CATEGORY]).tolist()
    unique_values_numbers = le.transform(unique_values_text)
    unique_values_numbers_one_hot = to_categorical(unique_values_numbers)

    # printing map of unique values, just because I can lol why not
    for i in range(len(unique_values_text)):
        print("{: 3} {: <10} ".format(unique_values_numbers[i], unique_values_text[i]),
              unique_values_numbers_one_hot[i])

    print(
        "Predicted category of image",
        image_path,
        "for category",
        CLASSIFICATION_CATEGORY,
        "is",
        le.inverse_transform([predicted_labels])[0])


# *******************************************************
# This function shows an image from array of obtained images
# This is just a check to control that we have written images to array
# in a correct form
# *******************************************************
def show_image_from_array(image):
    plt.imshow(image, cmap="gray")
    plt.show()


# ******************************************************************************
# This function creates a model of Convolution Network and returns its object
# ******************************************************************************
def get_convolutional_network_model(features_train=None, labels_train=None):
    # if trained model and its history are already saved - load them
    if os.path.isfile(os.path.join(CONV_NETWORK_MODEL_SAVED_DIR, CONV_NETWORK_MODEL_NAME)) \
            and os.path.isfile(os.path.join(CONV_NETWORK_MODEL_SAVED_DIR, CONV_NETWORK_FIT_HISTORY)):
        model = load_model(os.path.join(CONV_NETWORK_MODEL_SAVED_DIR, CONV_NETWORK_MODEL_NAME))
        with open(os.path.join(CONV_NETWORK_MODEL_SAVED_DIR, CONV_NETWORK_FIT_HISTORY), "rb") as f:
            history = keras.callbacks.History()
            history.history = json.load(f)
    # else create new ones
    else:
        # -----------------------------------------------------------------------------------
        # the first structure of the network
        # ***********************************************************************************
        # model = Sequential()
        #
        # model.add(Conv2D(256, (3, 3), input_shape=features_train.shape[1:]))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        #
        # model.add(Conv2D(256, (3, 3)))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        #
        # model.add(Flatten())
        # model.add(Dense(64))
        #
        # model.add(Dense(int(labels_train.shape[1])))
        # model.add(Activation('softmax'))
        # ----------------------------------------------------------------------------------

        # ----------------------------------------------------------------------------------------------------------
        # the second (better) structure of the network
        # **********************************************************************************************************
        model = Sequential()

        model.add(Conv2D(
                    32,
                    kernel_size=(3, 3),
                    activation='linear',
                    padding='same',
                    input_shape=features_train.shape[1:]))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D((2, 2), padding='same'))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), activation='linear', padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), activation='linear', padding='same'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
        model.add(Dropout(0.4))

        model.add(Flatten())
        model.add(Dense(128, activation='linear'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.3))

        model.add(Dense(int(labels_train.shape[1]), activation='softmax'))
        # -------------------------------------------------------------------------------------------------

        model.compile(
            loss="categorical_crossentropy",
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            metrics=['accuracy'])

        history = model.fit(
            features_train,
            labels_train,
            batch_size=64,
            validation_split=0.2,
            epochs=25,
            callbacks=[TensorBoard(log_dir='logs/{}'.format("conv-network-{}".format(int(time.time()))))],
            verbose=1)

        save_model(model, history.history)

    return model, history


# **************************************************************
# This function reads images from DATADIR directory
# **************************************************************
def obtain_images(df):
    training_data = []

    def create_training_data():
        path = DATADIR
        for img in os.listdir(path):
            # uncomment this condition if you want to limit your data
            # if len(training_data) > 2000:
            #     break
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                img_id = os.path.splitext(img)[0]
                row = df.loc[df['id'] == img_id]
                # if there is no record for this image - skip this looser ha ha lol
                if row.shape[0] == 0:
                    continue
                # if you would like to train the model for image classification by
                # another parameter (for example, masterCategory, usage, season) - just change the value of
                # CLASSIFICATION_CATEGORY to whatever you want and it will work without any problems
                # * of course classification category must be in the given dataset
                row = row[CLASSIFICATION_CATEGORY].values[0]
                training_data.append([new_array, row])
                print(len(training_data))
            except Exception as e:
                pass

    create_training_data()
    random.shuffle(training_data)

    features_list = []
    labels_list = []

    for features, label in training_data:
        features_list.append(features)
        labels_list.append(label)

    features_list = np.array(features_list).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    save_dataset(features_list, labels_list)
    return features_list, labels_list


# ********************************************************************************************
# This function creates a dataframe from csv file style.csv and returns it
# ********************************************************************************************
def get_dataframe(encode: bool = True):
    def encode_data(df):
        le = ppc.LabelEncoder()
        for column in df.columns:
            if column != 'id':
                df[column] = le.fit_transform(df[column])

    columns = [
        'id',
        'gender',
        'masterCategory',
        'subCategory',
        'articleType',
        'baseColour',
        'season',
        'year',
        'usage',
        'productDisplayName'
    ]
    trash = [
        'productDisplayName',
        'year',
        'baseColour',
        'subCategory',
        'articleType'
    ]
    data = pd.read_csv('styles.csv', header=None, names=columns)
    data.drop(trash, inplace=True, axis=1)
    data.dropna(inplace=True)
    data.drop_duplicates()
    data = data.iloc[1:]

    # TODO: pandas.get_dummies()

    if encode:
        encode_data(data)
        data.apply(pd.to_numeric)

    return data


# ********************************************************************************************
# This function prepares data for training, reads styles.csv and obtains images
# If started, continues about 7 minutes
# ********************************************************************************************
def prepare_data():
    data = get_dataframe()
    return obtain_images(data)


# ***********************************************
# This function trains a Convolution Network
# ***********************************************
def conv_net(X, y):
    # normalizing values so they are in range 0 - 1
    X = np.array(X / 255.0)
    # OneHot encoding, it is necessary
    y = to_categorical(np.array(y))

    features_train, features_test, labels_train, labels_test = train_test_split(X, y, test_size=0.33, random_state=3)
    model, history = get_convolutional_network_model(X, y)

    model.summary()

    evaluation = model.evaluate(features_test, labels_test, verbose=0)
    print('Test loss:', evaluation[0])
    print('Test accuracy:', evaluation[1])

    predictions = model.predict(features_test)
    predicted_labels = np.argmax(predictions, axis=1)

    labels = np.argmax(labels_test, axis=1)

    report = sklearn.metrics.classification_report(labels, predicted_labels)
    print("classification_report:\n", report)

    confusion_matrix = sklearn.metrics.confusion_matrix(y_true=labels, y_pred=predicted_labels)
    print("confusion_matrix:\n", confusion_matrix)

    visualize_filters(model)
    show_plot(history.history)


# ****************************************************************************************************
# This function returns existing datasets X and y if they exist or creates new ones if not
# ****************************************************************************************************
def get_X_y():
    if os.path.isfile("X.pickle") and os.path.isfile("y.pickle"):
        with open("X.pickle", "rb") as f:
            features = pickle.load(f)
        with open("y.pickle", "rb") as f:
            labels = pickle.load(open("y.pickle", "rb"))
    else:
        features, labels = prepare_data()

    return features, labels


# ****************************************************************************************************
# This function removes saved data, such as "X.pickle", "y.pickle",
# directory CONV_NETWORK_MODEL_SAVED_DIR with saved model and its training history
# Call this function if you want to clear data and begin training from the very beginning
# ****************************************************************************************************
def remove_saved_data():
    import shutil
    if os.path.isfile("X.pickle") and os.path.isfile("y.pickle"):
        os.remove("X.pickle")
        os.remove("y.pickle")
    if os.path.exists(CONV_NETWORK_MODEL_SAVED_DIR):
        shutil.rmtree(CONV_NETWORK_MODEL_SAVED_DIR)


if __name__ == '__main__':
    # uncomment next line if you want to remove saved data
    # # # # # # remove_saved_data()
    X, y = get_X_y()
    conv_net(X, y)
    test_conv_network_on_image('teniska.png')
