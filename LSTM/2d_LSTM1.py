from os import listdir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from keras import backend
from keras.layers import Dense, LSTM
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
    plt.savefig("{}-raw-data.png".format(stock))

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
    mean = train_array.mean(axis=0)
    std = train_array.std(axis=0)
    print(stock)
    print(mean)
    print(std)

    train_array = (train_array - mean) / std
    val_array = (data_array[num_train: (num_train + num_val)] - mean) / std
    test_array = (data_array[(num_train + num_val):] - mean) / std

    # print(stock)
    # print(f"train set size: {train_array.shape}")
    # print(f"validation set size: {val_array.shape}")
    # print(f"test set size: {test_array.shape}")

    return train_array, val_array, test_array


def create_tf_dataset(
        input_array: np.ndarray,
        target_array: np.ndarray,
        input_sequence_length: int,
        forecast_horizon: int,
        batch_size: int = 128
):
    """Creates tensorflow dataset from numpy array.

    This function creates a dataset where each element is a tuple `(inputs, targets)`.
    `inputs` is a Tensor
    of shape `(batch_size, input_sequence_length, num_routes, 1)` containing
    the `input_sequence_length` past values of the timeseries for each node.
    `targets` is a Tensor of shape `(batch_size, forecast_horizon, num_routes)`
    containing the `forecast_horizon`
    future values of the timeseries for each node.

    Args:
        data_array: np.ndarray with shape `(num_time_steps, num_routes)`
        input_sequence_length: Length of the input sequence (in number of timesteps).
        forecast_horizon: If `multi_horizon=True`, the target will be the values of the timeseries for 1 to
            `forecast_horizon` timesteps ahead. If `multi_horizon=False`, the target will be the value of the
            timeseries `forecast_horizon` steps ahead (only one value).
        batch_size: Number of timeseries samples in each batch.
        multi_horizon: See `forecast_horizon`.

    Returns:
        A tf.data.Dataset instance.
    """

    inputs = timeseries_dataset_from_array(
        input_array[:-forecast_horizon],
        None,
        sequence_length=input_sequence_length,
        shuffle=False,
        batch_size=batch_size,
    )

    targets = timeseries_dataset_from_array(
        (target_array[input_sequence_length:])[:, 0],
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
    model.add(LSTM(64, input_shape=(input_sequence_length, number_feature)))
    # model.add(LSTM(64, input_shape=(input_sequence_length, number_feature), return_sequences=True))
    # model.add(LSTM(64, return_sequences=True))
    # model.add(LSTM(32))
    model.add(Dense(1))
    model.summary()
    plot_model(model, to_file='LSTM-{}feature.png'.format(number_feature), show_shapes=True, show_layer_names=True)

    model.compile(loss='mae', optimizer='adam')
    model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

    return model

def plot_predicted_result(y, y_pred, stock):
    plt.figure(figsize=(18, 6))
    plt.plot(y[:, 0])
    plt.plot(y_pred[:, 0])
    plt.legend(["actual", "forecast"])
    plt.savefig(stock + "-predict-" + filepath + ".png")


def predict(model, test_dataset: np.ndarray, stock):
    x_test, y = next(test_dataset.as_numpy_iterator())
    print(x_test.shape)
    y_pred = model.predict(x_test)
    print(y_pred.shape)
    plot_predicted_result(y, y_pred, stock)

    # class_array_y_pred = y_pred[:, 0].astype(int)
    # class_array_y_pred = (array_y_pred[forecast_horizon:]/array_y_pred[:-forecast_horizon]).astype(int)
    test = y[:, 0]
    test_pred = y_pred[:, 0]
    metric_results = [mae(test_pred, test), np.sqrt(mse(test_pred, test))]
    # metric_results= metric_results+ [accuracy(test_pred, test), f1_score(test_pred, test, average='macro')]
    print("metric_results")
    print(metric_results)
    # DJI sin factor-> naive MAE: 0.0021673174502306954, model MAE: 0.060806224362133544
    # DJI range(1,10)-> naive MAE: 0.0021673174502306954, model MAE: 0.16460517128944463
    # DJI range(1,5)-> naive MAE: naive MAE: 0.0021673174502306954, model MAE: 0.2585225251668954
    # DJI range(0,1), all stocks-> nnaive MAE: 0.0021673174502306954, model MAE: 0.028034593440144702
    # DJI range(0,4), all stocks-> naive MAE: 0.0021673174502306954, model MAE: 0.05332288445767822
    # DJI range(0,1), all stocks-> [0.48148148148148145, 0.5185185185185185, 0.5105053881173285, 0.48148148148148145]
    # DJI range(0,4), all stocks-> [0.43434343434343436, 0.5656565656565656, 0.556793336803748, 0.43434343434343436]
    return metric_results

def create_bar_plot(results):
    labels = results.keys()
    print(np.array(list(results.values())))
    values=np.array(list(results.values()))
    print(values.shape)
    summary_means = values[:,0]
    print("summary_means")
    print(str(summary_means))
    base_means = values[:,1]
    print("base_means")
    print(str(base_means))

    # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, summary_means, width, label='MAE')
    ax.bar(x + width / 2, base_means, width, label='RMSE')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    print(summary_means + base_means)
    ax.set_yticks(np.arange(0, max([*summary_means, *base_means]) + 0.1, 0.1))
    # ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.savefig("results-"+filepath+".png")

train_size, val_size = 0.6, 0.2
features = list(range(0,1))#+[6,10,14,17,19,31,37,54,61,65]
# features = range(0, 1) #INPUT
print(features)
number_feature = len(features)
print(number_feature)
batch_size = 128
input_sequence_length = 60
forecast_horizon = 1
epochs = 200
add_all_datasets_data = True

filepath = "{}-{}-{}-{}-{}".format(
    input_sequence_length,
    number_feature,
    epochs,
    batch_size,
    add_all_datasets_data
)

data_files_path = "LSTM/Datasets"
data_files_names = listdir(data_files_path)
stocks = []
train_arrays = {}
val_arrays = {}
test_arrays = {}


for file in data_files_names:
    data = pd.read_csv("LSTM/Datasets/{}".format(file))
    stock = data['Name'][0]
    stocks.append(stock)
    del data["Name"]
    if number_feature >1:
        data.insert(2,"Day",data['Date'].apply(lambda dt_str: datetime.strptime(dt_str, '%Y-%m-%d').weekday()))
    print(data.head())
    del data['Date']
    print(data.values.shape)
    data_array = data.values[:, features]
    print(data_array.shape)
    data_array = data_array[200:]
    print(data_array.shape)
    print(data_array[:, 0])
    print("data_array[forecast_horizon:].shape")
    print(data_array[forecast_horizon:].shape)
    train_array, val_array, test_array = preprocess(data_array[forecast_horizon:], train_size, val_size, stock)


    train_array_target, val_array_target, test_array_target = train_array, val_array, test_array
    train_arrays[stock]=train_array

print("TRAIN & VAL")
train_dataset, val_dataset = (
    create_tf_dataset(data_array, target_array, input_sequence_length, forecast_horizon, batch_size)
    for (data_array, target_array) in [(train_array, train_array_target), (val_array, val_array_target)]
)
train_datasets[stock]=train_dataset
val_datasets[stock]=val_dataset
print("TESTTTT")
test_dataset = create_tf_dataset(
    test_array,
    test_array_target,
    input_sequence_length,
    forecast_horizon,
    batch_size=test_array.shape[0]
)
test_datasets[stock] = test_dataset





# results = {}
# print(stocks)
# if add_all_datasets_data:
#
#     x = tf.data.Dataset.from_tensor_slices(train_datasets.values())
#     concat_ds = x.interleave(lambda x: x, cycle_length=1, num_parallel_calls=tf.data.AUTOTUNE)
#
#     x = tf.data.Dataset.from_tensor_slices(val_datasets.values())
#     concat_ds1 = x.interleave(lambda x: x, cycle_length=1, num_parallel_calls=tf.data.AUTOTUNE)
#
#     for b in concat_ds:
#         i, t =b
#         print("input")
#         print(i.shape)
#         print("target")
#         print(t.shape)
#
#     print("--------------")
#     for b in concat_ds1:
#         i, t =b
#         print("input")
#         print(i.shape)
#         print("target")
#         print(t.shape)
#
#     model = LSTM_model(
#         concat_ds,
#         concat_ds1,
#         input_sequence_length,
#         number_feature,
#         epochs
#     )
#
#     for stock in stocks:
#         results[stock] = predict(model, test_datasets[stock], stock)
#
# else:
#
#     for stock in stocks:
#         model = LSTM_model(
#             train_datasets[stock],
#             val_datasets[stock],
#             input_sequence_length,
#             number_feature,
#             epochs
#         )
#
#         results[stock] = predict(model, test_datasets[stock], stock)
#
# print("results")
# print(results)
# create_bar_plot(results)
# pd.DataFrame.from_dict(results, orient='index', columns=["MAE", "RMSE"]).to_csv("results-"+filepath+".csv")