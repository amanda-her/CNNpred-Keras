from os import listdir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from keras import backend
from keras.layers import Dense, LSTM, RepeatVector, TimeDistributed, Conv1D, MaxPool1D
from keras.models import Sequential
from keras.utils import timeseries_dataset_from_array
from keras.utils.vis_utils import plot_model
from sklearn.metrics import accuracy_score as accuracy, f1_score, mean_absolute_error as mae, mean_squared_error as mse


def plot_raw_data(data_array, num_train, num_val, num_time_steps, stock):

    train_array = data_array[:num_train]
    val_array = data_array[num_train: (num_train + num_val)]
    test_array = data_array[(num_train + num_val):]

    plt.figure(figsize=(18, 6))
    plt.plot(range(0, num_train), train_array[:, 0])
    plt.plot(range(num_train, (num_train + num_val)), val_array[:, 0])
    plt.plot(range((num_train + num_val), num_time_steps), test_array[:, 0])
    plt.legend(["train", "validation", "test"])
    plt.title(stock)
    plt.savefig("TEST-{}-raw-data.png".format(stock))

    return


def preprocess(data_array: np.ndarray, train_size: float, val_size: float, stock: str):
    """Splits data into train/val/test sets and normalizes the data.

    Args:
        data_array: ndarray of shape `(num_time_steps, num_routes)`
        train_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset
            to include in the train split.
        val_size: A float value between 0.0 and 1.0 that represent the proportion of the dataset
            to include in the validation split.

    Returns:
        `train_array`, `val_array`, `test_array`
    """

    num_time_steps = data_array.shape[0]
    num_train, num_val = (
        int(num_time_steps * train_size),
        int(num_time_steps * val_size),
    )
    # plot_raw_data(data_array, num_train, num_val, num_time_steps, stock)
    train_array = data_array[:num_train]
    min = train_array.min(axis=0)
    max = train_array.max(axis=0)
    print(stock)
    print(min)
    print(max)

    train_array = (train_array - min) / (max-min)
    val_array = (data_array[num_train: (num_train + num_val)] - min) / (max-min)
    test_array = (data_array[(num_train + num_val):] - min) / (max-min)

    # print(stock)
    # print(f"train set size: {train_array.shape}")
    # print(f"validation set size: {val_array.shape}")
    # print(f"test set size: {test_array.shape}")

    return train_array, val_array, test_array, min, max


def create_tf_dataset(
        input_array: np.ndarray,
        input_sequence_length: int,
        forecast_horizon: int,
        batch_size: int = 128
):
    print("input_array.shape")
    print(input_array.shape)
    print("input_array[:-forecast_horizon].shape")
    print(input_array[:-forecast_horizon].shape)
    inputs = timeseries_dataset_from_array(
        input_array[:-forecast_horizon],
        None,
        sequence_length=input_sequence_length,
        shuffle=False,
        batch_size=batch_size,
    )

    print("input_array[input_sequence_length:].shape")
    print(input_array[input_sequence_length:].shape)
    targets = timeseries_dataset_from_array(
        input_array[input_sequence_length:],
        None,
        sequence_length=1,
        shuffle=False,
        batch_size=batch_size,
    )

    dataset = tf.data.Dataset.zip((inputs, targets))
    print("---------------")
    for batch in dataset:
        inputs, targets = batch
        print(inputs.shape)
        print(targets.shape)
    print("=========================")
    print(type(dataset))
    print(dataset._variant_tensor)
    print("***************************")

    return dataset.prefetch(16).cache()


def LSTM_model(
        train_dataset: np.ndarray,
        val_dataset: np.ndarray,
        input_sequence_length: int,
        number_feature: int,
        epochs: int
):
    backend.clear_session()
    model = Sequential()
    model.add(LSTM(256, input_shape=(input_sequence_length, number_feature)))
    model.add(Dense(number_feature))
    model.summary()
    plot_model(model, to_file='{}-{}feature.png'.format(model_input, number_feature), show_shapes=True, show_layer_names=True)

    model.compile(loss='mae', optimizer='adam')
    model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

    return model

def CNN_LSTM_model(
        train_dataset: np.ndarray,
        val_dataset: np.ndarray,
        input_sequence_length: int,
        number_feature: int,
        epochs: int
):
    backend.clear_session()
    model = Sequential()
    model.add(Conv1D(64, kernel_size=8, input_shape=(input_sequence_length, number_feature)))
    # model.add(MaxPool1D(pool_size=1))
    # model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(256))
    #[0.06408113 0.03735174 0.12168183 0.11886682 0.13599887] 3,256
    model.add(Dense(number_feature))
    model.summary()
    plot_model(model, to_file='{}-{}feature.png'.format(model_input, number_feature), show_shapes=True, show_layer_names=True)

    model.compile(loss='mae', optimizer='adam')
    model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

    return model

def plot_predicted_result(y, y_pred, stock, axes, i, mean, std):
    # plt.figure(figsize=(18, 6))
    axes[i].plot(y*(std-mean)+mean, label="{} actual".format(stock))
    axes[i].plot(y_pred*(std-mean)+mean,  label="{} forecast".format(stock))
    axes[i].legend()
    # plt.title(stock)
    # plt.savefig(stock + "-predict-" + filepath + ".png")


def predict(model, test_dataset: np.ndarray, means, stds, train_array, val_array):
    x_test, y = next(test_dataset.as_numpy_iterator())
    print(x_test.shape)
    print(y.shape)
    y_pred = model.predict(x_test)
    print(y_pred.shape)
    # plot_predicted_result(y, y_pred, stock)


    test = np.squeeze(y)
    test_pred = np.squeeze(y_pred)
    print(test.shape)
    print(test_pred.shape)
    print(test_pred)
    print(test_pred[:,4])


    metric_results=[]
    fig, axs = plt.subplots(len(stocks), sharex=True,  figsize=(16, 16))
    fig.suptitle('Sharing both axes')
    for i in range(0,5):
        print(i)
        print(test[:,i])
        print(test_pred[:,i])
        plot_predicted_result(test[:,i], test_pred[:,i], stocks[i], axs, i, means[i], stds[i])
        print(abs((test[:, i]-test_pred[:,i])).mean())
        metric_results.append([mae(test_pred[:,i], test[:,i]), np.sqrt(mse(test_pred[:,i], test[:,i]))])
    for ax in axs:
        ax.label_outer()
    plt.savefig("predict-" + filepath + ".png")



    print("metric_results")
    print(metric_results)
    return metric_results

def create_bar_plot(results):
    labels = stocks
    summary_means = results[:,0]
    print("summary_means")
    print(str(summary_means))
    base_means = results[:,1]
    print("base_means")
    print(str(base_means))

    # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    ax.grid(True, axis='y', zorder=0)
    ax.bar(x - width / 2, summary_means, width, label='MAE', zorder=10)
    ax.bar(x + width / 2, base_means, width, label='RMSE', zorder=10)

    # Add some text for labels, title and custom x-axis tick labels, etc.
    print(summary_means + base_means)
    ax.set_yticks(np.arange(0, max([*summary_means, *base_means]) + 0.05, 0.05))
    # ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.savefig("results-"+filepath+".png")

train_size, val_size = 0.6, 0.2
features = [0]
number_feature = 5
batch_size = 128
input_sequence_length = 10
forecast_horizon = 1
epochs = 50
model_input="LSTM" #LSTM

filepath = "DIFF-{}-{}-{}-{}-{}-".format(
    model_input,
    input_sequence_length,
    number_feature,
    epochs,
    batch_size
)

data_files_path = "LSTM/Datasets"
data_files_names = listdir(data_files_path)
stocks = []
train_arrays = []
val_arrays = []
test_arrays = []
means = []
stds = []

i=0
for file in data_files_names:
    data = pd.read_csv("LSTM/Datasets/{}".format(file))
    stock = data['Name'][0]
    print(stock)
    stocks.append(stock)
    del data["Name"]
    del data['Date']
    print(data.values.shape)
    data_array = data.values[:, features]
    print(data_array.shape)
    data_array = data_array[200:]
    print(data_array.shape)
    print(data_array[:, 0])
    print("data_array[forecast_horizon:].shape")
    print(data_array[forecast_horizon:].shape)
    train_array, val_array, test_array, mean, std = preprocess(data_array[forecast_horizon:], train_size, val_size, stock)
    means.append(mean)
    stds.append(std)
    train_arrays.append(train_array)
    val_arrays.append(val_array)
    test_arrays.append(test_array)
    print(train_array.shape)
    print(train_array)
    i+=1


train_array=np.squeeze(np.array(train_arrays).T)
print(train_array.shape)
print(train_array)
val_array=np.squeeze(np.array(val_arrays).T)
print(val_array.shape)
test_array=np.squeeze(np.array(test_arrays).T)
print(test_array.shape)

print("TRAIN & VAL")
train_dataset, val_dataset = (
    create_tf_dataset(data_array, input_sequence_length, forecast_horizon, batch_size)
    for data_array in [train_array, val_array]
)

test_dataset = create_tf_dataset(
    test_array,
    input_sequence_length,
    forecast_horizon,
    batch_size=test_array.shape[0]
)

results = {}
if model_input=="CNN-LSTM":
    model = CNN_LSTM_model(
        train_dataset,
        val_dataset,
        input_sequence_length,
        number_feature,
        epochs
    )
else:
    model = LSTM_model(
        train_dataset,
        val_dataset,
        input_sequence_length,
        number_feature,
        epochs
    )

results = predict(model, test_dataset, means, stds, train_array, val_array)

print("results")
print(results)
create_bar_plot(np.array(results))
x=np.append(np.array(stocks)[..., None], np.array(results), axis=1)
pd.DataFrame(x,  columns=["stock", "MAE", "RMSE"]).to_csv("results-"+filepath+".csv", index=False)
