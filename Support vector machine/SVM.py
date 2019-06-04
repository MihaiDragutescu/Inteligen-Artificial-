import csv

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import preprocessing
import sklearn.metrics as sm
import pdb
import os.path
from sklearn.model_selection import KFold


def normalize_data(train_data, test_data, type=None):
    if type is None:
        return train_data, test_data

    if (type == 'standard'):
        scaler = preprocessing.StandardScaler()
        scaler.fit(train_data)
        train_data = scaler.transform(train_data)
        test_data = scaler.transform(test_data)
        return train_data, test_data

    if (type == 'min_max'):
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
        min_max_scaler.fit(train_data)
        train_data = min_max_scaler.transform(train_data)
        test_data = min_max_scaler.transform(test_data)
        return train_data, test_data

    if (type == 'l1'):
        train_data /= np.sum(abs(train_data), axis=1, keepdims=True)
        test_data /= np.sum(abs(test_data), axis=1, keepdims=True)
        return train_data, test_data

    if (type == 'l2'):
        train_data /= np.sqrt(np.sum((train_data) ** 2, axis=1, keepdims=True))
        test_data /= np.sqrt(np.sum((test_data) ** 2, axis=1, keepdims=True))
        return train_data, test_data


def svm_classifier(train_data, train_labels, test_data, c):
    linear_svm_model = svm.SVC(C=c, kernel='rbf')
    linear_svm_model.fit(train_data, train_labels.ravel())
    predicted_labels_train = linear_svm_model.predict(train_data)
    predicted_labels_test = linear_svm_model.predict(test_data)
    return predicted_labels_train, predicted_labels_test


if __name__ == '__main__':

    train_data = np.zeros(shape=(9000, 450))
    index = 0

    # salvam intr-o lista numele fisierelor
    nume_fisiere = []

    # incarcam datele de train
    for i in range(10003, 24000):
        # train_data_array = np.zeros(shape=(0, 450))
        file_path = 'train/' + str(i) + '.csv'

        # daca exista fisierul la calea "file_path", incarcam datele din acesta intr-un NumPy array si salvam numele fisierului
        if (os.path.isfile(file_path)):
            train_data_file = np.loadtxt(file_path, delimiter=',')

            # convertim la NumPy 1D array
            train_data_array = np.ravel(train_data_file).copy()

            # daca fisierul continea mai mult de 150 linii (implicit mai mult de 150*3=450 elemente), eliminam datele in plus
            if len(train_data_array) > 450:
                train_data_array.resize(450)

            # daca fisierul continea mai putin de 150 linii, adaugam valori de 0 pentru a completa pana la al 450-lea element cu 0
            elif len(train_data_array.shape) < 450:
                for j in range(train_data_array.shape[0], 450):
                    train_data_array = np.append(train_data_array, 0)
            else:
                continue

            # adaugam datele din fisierul curent la datele de train
            train_data[index] = train_data_array
            index += 1
            nume_fisiere.append(index)

    test_data = np.zeros(shape=(5000, 450))
    index = 0
    nume_fisiere = []

    # incarcam datele de test procedand similar
    for i in range(10001, 24001):
        file_path = 'test/' + str(i) + '.csv'
        if (os.path.isfile(file_path)):
            test_data_file = np.loadtxt(file_path, delimiter=',')
            test_data_array = np.ravel(test_data_file).copy()
            nume_fisiere.append(i)

            if len(test_data_array) > 450:
                test_data_array.resize(450)

            elif len(test_data_array.shape) < 450:
                for j in range(test_data_array.shape[0], 450):
                    test_data_array = np.append(test_data_array, 0)
            else:
                continue

            test_data[index] = test_data_array
            index += 1

    # incarcam etichetele de train
    labels = np.loadtxt('train_labels.csv', delimiter=',', skiprows=1)
    labels = np.delete(labels, 0, 1)

    scaled_training_data, scaled_test_data = normalize_data(train_data, test_data, type='standard')
    predicted_labels_train, predicted_labels_test = svm_classifier(scaled_training_data, labels, scaled_test_data, 6.5)
    predicted_labels_test = predicted_labels_test.astype(int)

    # scriem intr-un fisier CSV predictiile obtinute
    with open("predicted_labels.csv", "w", newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(["id", "class"])
        for i in range(0, len(predicted_labels_test)):
            writer.writerow([nume_fisiere[i], predicted_labels_test[i]])
