import numpy as np
import os

from sklearn.metrics import fbeta_score
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from  keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras.callbacks import Callback
from keras.applications import Xception
from tensorflow.contrib.keras import backend as K


def fbeta_bce_loss(y_true, y_pred, beta=2):
    beta_sq = beta ** 2
    tp_loss = K.sum(y_true * (1 - K.binary_crossentropy(y_pred, y_true)), axis=-1)
    fp_loss = K.sum((1 - y_true) * K.binary_crossentropy(y_pred, y_true), axis=-1)

    return K.mean((1 + beta_sq) * tp_loss / ((beta_sq * K.sum(y_true, axis=-1)) + tp_loss + fp_loss))


class LossHistory(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.train_acc = []
        self.val_acc = []

    def on_epoch_end(self, epoch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.train_acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))

class XceptionClassifier:
    def __init__(self, image_size=74):
        self.losses = []
        self.classifier = Sequential()
        conv_base = Xception(include_top=False, weights='imagenet', input_tensor=None,
                             input_shape=(image_size, image_size,3), pooling='avg')
        conv_base.trainable = False
        self.classifier.add(conv_base)

    def add_classification(self, output_size):
#        self.classifier.add(GlobalAveragePooling2D())
        self.classifier.add(Dense(256, activation='relu'))
        self.classifier.add(Dropout(0.5))
        self.classifier.add(Dense(512, activation='relu'))
        self.classifier.add(Dropout(0.5))
        self.classifier.add(Dense(output_size, activation='sigmoid'))

    def _get_fbeta_score(self, classifier, X_valid, y_valid):
        p_valid = classifier.predict(X_valid)
        return fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples')

    def train_model(self, x_train, y_train, epoch=5, batch_size=128, validation_split_size=0.2, train_callbacks=()):
        history = LossHistory()

        X_train, X_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=validation_split_size)

        self.classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        self.classifier.fit(X_train, y_train, batch_size=batch_size, epochs=epoch, verbose=1,
                            validation_data=(X_valid, y_valid), callbacks=[history, *train_callbacks])

        fbeta_score = self._get_fbeta_score(self.classifier, X_valid, y_valid)

        return [history.train_losses, history.val_losses, history.train_acc, history.val_acc, fbeta_score]

    def predict(self, x_test):
        predictions = self.classifier.predict(x_test)
        return predictions

    def map_predictions(self, predictions, labels_map, thresholds):
        """
        Return the predictions mapped to their labels
        :param predictions: the predictions from the predict() method
        :param labels_map: the map
        :param thresholds: The threshold of each class to be considered as existing or not existing
        :return: the predictions list mapped to their labels
        """
        predictions_labels = []
        for prediction in predictions:
            labels = [labels_map[i] for i, value in enumerate(prediction) if value > thresholds[i]]
            predictions_labels.append(labels)

        return predictions_labels

    def close(self):
        K.clear_session()

