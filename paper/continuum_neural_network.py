# -*- coding: utf-8 -*-
""" Minimum working example of an SME script
"""
import os.path
import pickle
import time

import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.constraints import max_norm, unit_norm
from keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    MaxPooling1D,
)
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Input, Model, Sequential, load_model
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import make_friedman1, make_regression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from pysme import sme as SME
from pysme import util
from pysme.abund import Abund
from pysme.gui import plot_plotly
from pysme.linelist.vald import ValdFile
from pysme.persistence import save_as_idl
from pysme.solve import solve
from pysme.synthesize import synthesize_spectrum


def create_sme_structure(teff=5770, logg=4.4):
    examples_dir = os.path.dirname(os.path.realpath(__file__))
    in_file = os.path.join(examples_dir, "sun_6440_grid.inp")
    # Load your existing SME structure or create your own
    sme = SME.SME_Structure.load(in_file)
    sme.abund = Abund(0.05, "asplund2009")
    sme.linelist = ValdFile(os.path.join(examples_dir, "sun.lin"))
    # Change parameters if your want
    sme.vrad_flag = "none"
    sme.cscale_flag = "none"
    sme.cscale_type = "mask"
    sme.vsini = 0
    sme.vrad = 0
    # Set input parameters
    sme.teff = teff
    sme.logg = logg
    # Start SME solver
    sme = synthesize_spectrum(sme)
    return sme


def create_training_data(sme, n_samples=100, max_degree=10, noise=0.01):
    spec = sme.synth[0]
    # The mask is defined by the points that are unaffected by lines, or at least not much
    # This makes mask a classifier with the values 0 and 1
    mask = sme.synth[0] > 0.99
    mask = np.where(mask, 1, 0)

    n_features = len(spec)
    # modulate the observation to generate training data
    X = np.linspace(-1, 1, n_features)
    rng = np.random.default_rng()

    degree = rng.integers(0, max_degree + 1, size=n_samples)

    coeff = np.zeros((n_samples, max_degree + 1))
    for i in range(max_degree + 1):
        coeff[degree >= i, -1 - i] = rng.standard_normal(
            size=np.count_nonzero(degree >= i)
        )

    P = np.zeros((n_samples, n_features))
    for i in range(n_samples):
        P[i] = np.polyval(coeff[i], X)

    P -= P.min(axis=1)[:, None]
    P /= P.max(axis=1)[:, None]
    P += 0.5
    P = np.nan_to_num(P, nan=1)

    for i in range(n_samples):
        coeff[i] = np.polyfit(X, P[i], deg=max_degree)

    # The output is always the normalized spectrum
    # TODO: or should the output be the continuum polynomial
    X = np.zeros_like(P)
    Y = np.zeros_like(P)
    shift = rng.integers(0, n_features, size=n_samples)
    flip = rng.integers(0, 2, size=n_samples) == 1
    noise = P * noise * rng.random(size=P.shape)
    for i in range(n_samples):
        X[i] = np.roll(spec, shift[i])
        Y[i] = np.roll(mask, shift[i])
    X[flip] = X[flip, ::-1]
    Y[flip] = Y[flip, ::-1]
    X *= P
    X += noise

    return X, Y


class ContinuumModel:
    def __init__(
        self,
        batch_size=64,
        epochs=200,
        activation="relu",
        optimizer="adam",
        loss="mse",
        verbose=False,
    ):
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.activation = activation
        self.optimizer = optimizer
        self.model = None
        self.scaler = StandardScaler()
        self.history = None

        if loss in ["mse", "mean_squared_error"]:
            self.loss = "mean_squared_error"
        else:
            self.loss = loss

    def create_model(self, input_shape, output_shape):
        input_shape = input_shape[1]
        output_shape = output_shape[1]

        self.model = Sequential()
        # self.model.add(Flatten(input_dim=input_shape))
        self.model.add(
            Conv1D(
                8,
                16,
                padding="same",
                activation=self.activation,
                input_shape=(1, input_shape),
            )
        )
        # self.model.add(Conv1D(32, 8, activation=self.activation))
        self.model.add(Flatten())
        self.model.add(Dense(32, activation=self.activation))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(16, activation=self.activation))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(32, activation=self.activation))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(output_shape, activation="sigmoid"))
        return self.model

    def fit(self, X, y, validation_data=None):
        self.create_model(X.shape, y.shape)
        self.model.compile(
            loss=self.loss, optimizer=self.optimizer, metrics=[self.loss]
        )

        X = self.scaler.fit_transform(X)
        X = X[:, None, ...]
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val = self.scaler.transform(X_val)
            validation_data = (X_val[:, None, ...], y_val)

        self.history = self.model.fit(
            X,
            y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=self.verbose,
            validation_data=validation_data,
        )
        return self.history

    def score(self, X, y):
        X = self.scaler.transform(X)
        X = X[:, None, ...]
        return self.model.evaluate(X, y, verbose=0)

    def predict(self, X):
        X = self.scaler.transform(X)
        X = X[:, None, ...]
        return self.model.predict(X)

    def save(self, filename):
        self.model.save(filename)
        with open(filename + "_", "wb") as f:
            pickle.dump(self.scaler, f)

    def load(self, filename):
        self.model = load_model(filename)
        with open(filename + "_", "rb") as f:
            self.scaler = pickle.load(f)


if __name__ == "__main__":
    max_degree = 5
    # int: samples per parameter setting
    n_samples = 5000
    # n_features = 1000
    noise = 0.05
    util.start_logging("neural_network.log")

    # TODO: Need to figure out how to handle different input sizes
    # interpolate, to a reasonable resolution
    # then run the thing for segments of size x

    start = time.time()
    # X, Y = None, None
    # for teff in [5500, 6000, 6500]:
    #     for logg in [4, 4.2, 4.4, 4.6]:
    #         for noise in [0.01, 0.05, 0.1]:
    #             sme = create_sme_structure(teff, logg)
    #             X2, Y2 = create_training_data(
    #                 sme, n_samples, max_degree=max_degree, noise=noise
    #             )
    #             X = np.concatenate((X, X2)) if X is not None else X2
    #             Y = np.concatenate((Y, Y2)) if Y is not None else Y2

    # np.savez("data.npz", X=X, Y=Y)
    sme = create_sme_structure()
    data = np.load("data.npz")
    X, Y = data["X"], data["Y"]

    # Split the data between training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y)
    X_train, X_validate, y_train, y_validate = train_test_split(
        X_train, y_train, test_size=0.1
    )

    # clf = MLPClassifier(hidden_layer_sizes=(128, 64, 32, 16), verbose=True)
    # clf = MLPRegressor(hidden_layer_sizes=(128, 64, 32, 32), verbose=True)
    # clf = PLSRegression(max_degree + 1)
    # clf.fit(XS_train, y_train)
    clf = ContinuumModel(verbose=True, loss="mse")
    history = clf.fit(X_train, y_train, validation_data=(X_validate, y_validate))

    clf.model.save("model.dat")

    # Predict coefficients
    y_predict = clf.predict(X_test)

    end = time.time()
    print(f"Runtime: {end - start} s")

    mse = history.history["loss"]
    val_mse = history.history["val_loss"]
    epochs = np.arange(len(mse))

    plt.plot(epochs, mse, "bo", label="Training MSE")
    plt.plot(epochs, val_mse, "b", label="Validation MSE")
    plt.title("Training and validation MSE")
    plt.legend()
    plt.show()

    # Plot comparison
    n_features = X.shape[1]
    x = np.linspace(-1, 1, n_features)

    ys_predict = clf.predict(sme.spec[0][None, :])
    mask_predict = ys_predict[0] > 0.5
    p_predict = np.polyfit(x[mask_predict], sme.spec[0][mask_predict], max_degree)
    ysp_predict = np.polyval(p_predict, x)

    plt.plot(x, ysp_predict, label="polynomial")
    plt.plot(x, sme.spec[0], label="original")
    plt.plot(x, sme.synth[0], label="expected")
    plt.plot(x, sme.spec[0] / ysp_predict, label="predicted")
    plt.plot(x[mask_predict], sme.spec[0][mask_predict], "+", label="mask")
    plt.legend()
    plt.show()

    for i in range(n_samples):
        mask_predict = y_predict[i] > 0.5
        p_predict = np.polyfit(x[mask_predict], X_test[i][mask_predict], max_degree)
        yp_predict = np.polyval(p_predict, x)

        mask_test = y_test[i] > 0.5
        p_test = np.polyfit(x[mask_test], X_test[i][mask_test], max_degree)
        yp_test = np.polyval(p_test, x)

        # plt.plot(x, yp_predict, label="prediction")
        # plt.plot(x, yp_test, label="expectation")
        # plt.plot(x[mask_predict], X_test[i][mask_predict], "+", label="mask")
        # plt.legend()
        # plt.show()

        plt.plot(x, sme.synth[0], label="original")
        plt.plot(x, X_test[i] / yp_test, label="expected")
        plt.plot(x, X_test[i] / yp_predict, label="predicted")
        plt.plot(
            x[mask_test],
            (X_test[i] / yp_test)[mask_test],
            "+",
            label="mask expected",
        )
        plt.plot(
            x[mask_predict],
            (X_test[i] / yp_predict)[mask_predict],
            "+",
            label="mask predicted",
        )
        plt.legend()
        plt.show()
    pass
