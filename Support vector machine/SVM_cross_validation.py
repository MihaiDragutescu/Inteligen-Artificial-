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

    labels = np.loadtxt('train_labels.csv', delimiter=',', skiprows=1)
    labels = np.delete(labels, 0, 1)

    # salvam intr-o lista numele fisierelor
    nume_fisiere = []

    for i in range(10003, 24000):
        file_path = 'train/' + str(i) + '.csv'

        # daca exista fisierul la calea "file_path", salvam numele fisierului intr-o lista
        if (os.path.isfile(file_path)):
            nume_fisiere.append(i)

    nume_fisiere = np.array(nume_fisiere)

    # urmeaza sa aplicam 3-fold cross validation pe datele din folder-ul "train" si pentru etichetele din fisierul "train_labels.csv"
    kf = KFold(n_splits=3, random_state=None, shuffle=False)

    for train_index, test_index in kf.split(nume_fisiere):

        labels_train = []
        labels_test = []

        train_data = np.zeros(shape=(6000, 450))
        test_data = np.zeros(shape=(3000, 450))
        index = 0

        # incarcam datele si etichetele de train
        for i in train_index:
            nr = nume_fisiere[i]
            file_path = 'train/' + str(nr) + '.csv'

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

            # adaugam datele la etichetele de train
            labels_train.append(labels[i])
        labels_train_array = np.array(labels_train)

        index = 0

        # incarcam datele si etichetele de test procedand similar
        for i in test_index:
            nr = nume_fisiere[i]
            file_path = 'train/' + str(nr) + '.csv'

            test_data_file = np.loadtxt(file_path, delimiter=',')
            test_data_array = np.ravel(test_data_file).copy()

            if len(test_data_array) > 450:
                test_data_array.resize(450)

            elif len(test_data_array.shape) < 450:
                for j in range(test_data_array.shape[0], 450):
                    test_data_array = np.append(test_data_array, 0)
            else:
                continue

            test_data[index] = test_data_array
            index += 1

            labels_test.append(labels[i])
        labels_test_array = np.array(labels_test)

        scaled_training_data, scaled_test_data = normalize_data(train_data, test_data, type='standard')
        predicted_labels_train, predicted_labels_test = svm_classifier(scaled_training_data, labels_train_array,
                                                                       scaled_test_data, 6.5)
        predicted_labels_test = predicted_labels_test.astype(int)
        print("Training accuracy: ",
              sm.accuracy_score(predicted_labels_train, labels_train_array))
        print("Test accuracy: ", sm.accuracy_score(predicted_labels_test, labels_test_array))

        print("Matricea de confuzie:")
        conf_mat = sm.confusion_matrix(labels_test_array, predicted_labels_test)
        print(conf_mat)
