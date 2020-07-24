from sklearn import svm, metrics
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os

x_train = []
y_train = []
x_test = []
y_test = []


def readImagesFromFile():
    global x_train, y_train, x_test, t_test
    for root, directory, files in os.walk("images/train"):
        for image in files:
            if '.jpg' in image:
                label = image.split('_')[0]  # assuming our img is named like this "0_1.jpg" we want to get the label "0"
                y_train.append(label)
                img = imageio.imread("images/train/" + image)
                color_features = img.flatten()
                image_features = np.hstack(color_features)
                x_train.append(image_features)
    for root, directory, files in os.walk("images/test"):
        for image in files:
            if '.jpg' in image:
                label = image.split('_')[0]  # assuming our img is named like this "0_1.jpg" we want to get the label "0"
                y_test.append(label)
                img = plt.imread("images/test/" + image)
                color_features = img.flatten()
                image_features = np.hstack(color_features)
                x_test.append(image_features)


def predict():
    classifier = svm.SVC(kernel='rbf')
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)

    print("Classification report for classifier %s:\n%s\n" % (classifier, metrics.classification_report(y_test, prediction)))
    display = metrics.plot_confusion_matrix(classifier, x_test, y_test, cmap="plasma")
    display.figure_.suptitle("Confusion Matrix")
    print("Confusion matrix:\n%s" % display.confusion_matrix)

    plt.show()


def startClassification():
    readImagesFromFile()
    predict()
