import numpy as np
import matplotlib.pyplot as plt
import random


class MultivariateTimeseriesBatchGenerator(object):
    """

    This class provides a generator for observations with different history length of time-series for neural network
    (LSTM or ) training

    Parameters of constructor:
    data: dataframe with model features and targets
    feature_count: count of features
    target_name: name of target column
    client_identificator: name of unique client identificator column
    max_history_length: maximum possible amount of timesteps
    """

    def __init__(self, data, feature_count, target_name, client_identificator, max_history_length):
        self.data = data
        self.feature_count = feature_count
        self.target_name = target_name
        self.client_identificator = client_identificator
        self.batch_size = 1
        self.current_idx = 0
        self.skip_step = \
        data[data[client_identificator] == data.iloc[0][data.columns.get_loc(client_identificator)]].shape[0]
        self.max_history_length = max_history_length

    def generate(self):
        while True:
            if self.current_idx + self.max_history_length >= len(self.data):
                self.current_idx = 0
            self.skip_step = self.data[self.data[self.client_identificator] == self.data.iloc[self.current_idx][
                self.data.columns.get_loc(self.client_identificator)]].shape[0]
            self.current_idx += self.skip_step
            x = np.zeros((self.batch_size, self.skip_step, self.feature_count))
            y = np.zeros((self.batch_size, 1))
            for i in range(self.batch_size):
                # set_trace()
                x[i, :] = self.data.iloc[self.current_idx - self.skip_step:self.current_idx, : self.feature_count]
                y[i, 0] = self.data.iloc[self.current_idx - self.skip_step][self.data.columns.get_loc(self.target_name)]
                # set_trace()
            yield x, y  # , self.current_idx, self.current_idx -self.skip_step


def shuffle(df):
    """
    This function shuffles a dataframe
    Parameters:
    df = Dataframe to be shuffled
    """
    index = list(df.index)
    random.shuffle(index)
    df = df.iloc[index]
    df.reset_index()
    return df


def train_test_split(df, ratio):
    """
    This function shuffles given dataframe and splits it to train and test set with given ratio
    Parameters:
    df: dataframe to split
    ratio: floating point number greater than 0 and less than 1 that represents train part proportion
    """
    lenght = df.shape[0]
    df_shuffled = shuffle(df)
    point = int(lenght * ratio)
    train = df_shuffled.iloc[:point, :]
    train = train.reset_index(drop=True)
    test = df_shuffled.iloc[point:, :]
    test = test.reset_index(drop=True)
    return train, test


def plot_nn_training(train_metric, val_metric, epochs, labels=0):
    """
    Plots a graph how training and validation metrics are changing during different epochs
    Parameters:
    train_metric: array of values of metric on training data during epochs
    val_metric: array of values of metric on validation data during epochs
    labels: names of axes: 0 - for loss metric, 1 - for accuracy metric
    """
    plt.clf()
    plt.figure(figsize=(18, 10))
    if labels == 0:
        label_t = 'Training loss'
        label_v = 'Validation loss'
        label_m = 'loss'
    else:
        label_t = 'Training acc'
        label_v = 'Validation acc'
        label_m = 'accuracy'
    plt.plot(epochs, train_metric, 'bo', label=label_t)
    plt.plot(epochs, val_metric, 'g', label=label_v)
    plt.title('Training and validation ' + label_m)
    plt.xlabel('Epochs')
    plt.ylabel(label_m.title())
    plt.xticks(epochs)
    plt.legend()
    plt.show()
