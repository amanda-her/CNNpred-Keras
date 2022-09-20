from os import listdir

import pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from keras import backend
from keras.layers import Dense, LSTM, Conv1D, MaxPool1D, Flatten, Dropout, Input, Permute, BatchNormalization, GlobalAveragePooling1D, concatenate, Activation
from keras.models import Sequential, Model
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
    min = train_array.min(axis=0)
    max = train_array.max(axis=0)
    print(stock)
    print(min)
    print(max)
    print(data_array[:24])
    print(num_time_steps)
    print(num_train)
    print(num_val)

    train_array = (train_array - min) / (max-min)
    val_array = (data_array[num_train: (num_train + num_val)] - min) / (max-min)
    test_array = (data_array[(num_train + num_val):] - min) / (max-min)

    # print(stock)
    # print(f"train set size: {train_array.shape}")
    # print(f"validation set size: {val_array.shape}")
    # print(f"test set size: {test_array.shape}")

    return train_array, val_array, test_array, np.array(min)[0], np.array(max)[0]


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
    # for batch in dataset:
    #     inputs, targets = batch
    #     print(inputs.shape)
    #     print(targets.shape)

    # print(type(dataset))
    # print(dataset._variant_tensor)

    return dataset.prefetch(16).cache()

def FCN_LSTM_model(
        train_dataset: np.ndarray,
        val_dataset: np.ndarray,
        input_sequence_length: int,
        number_feature: int,
        epochs: int
):
    input_shape = (input_sequence_length, number_feature)
    input_layer = Input(input_shape)

    # perm_layer = Permute((2, 1))(input_layer)
    lstm_layer = LSTM(128)(input_layer)
    lstm_layer = Dense(8)(lstm_layer)
    # lstm_layer = Dropout(0.8)(lstm_layer)

    conv1 = Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
    conv1 = BatchNormalization()(conv1)
    # conv1 = Activation(activation='relu')(conv1)

    conv2 = Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    # conv2 = Activation('relu')(conv2)

    conv3 = Conv1D(128, kernel_size=3, padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    # conv3 = Activation('relu')(conv3)

    gap_layer = GlobalAveragePooling1D()(conv3)

    gap_layer = Dense(8)(gap_layer)


    concat = concatenate([gap_layer, lstm_layer])


    output_layer = Dense(1)(concat)

    model = Model(inputs=input_layer, outputs=output_layer)
    plot_model(model, to_file='{}-{},{}-{}feature.png'.format(model_input, filter, units, number_feature),
               show_shapes=True, show_layer_names=True)

    model.compile(loss='mae', optimizer='adam')
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

    # list all data in history
    print(history.history.keys())
    # summarize history for loss
    # pl.figure(2)
    plt.figure(global_stock)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("Loss-{}-{}.png".format(filepath, global_stock))

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
    model.add(Conv1D(filter, kernel_size=1, input_shape=(input_sequence_length, number_feature)))
    model.add(MaxPool1D())
    model.add(Conv1D(filter, kernel_size=3))
    model.add(MaxPool1D())
    # model.add(LSTM(100, return_sequences=True))
    model.add(LSTM(units))
    #[0.06408113 0.03735174 0.12168183 0.11886682 0.13599887] 3,256
    model.add(Dense(1))
    model.summary()
    plot_model(model, to_file='{}-version2-{},{}-{}feature.png'.format(model_input, filter, units,number_feature), show_shapes=True, show_layer_names=True)

    model.compile(loss='mae', optimizer='adam')
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)


    # list all data in history
    print(history.history.keys())
    # summarize history for loss
    # pl.figure(2)
    plt.figure(global_stock)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("Loss-{}-{}.png".format(filepath, global_stock))
    # pl.figure(1)

    return model


def CNNpred_model(
        train_dataset: np.ndarray,
        val_dataset: np.ndarray,
        input_sequence_length: int,
        number_feature: int,
        epochs: int
):
    backend.clear_session()
    model = Sequential()
    # layer 1
    model.add(
        Conv1D(8, 1,  input_shape=(input_sequence_length, number_feature))
    )
    model.add(MaxPool1D())
    # layer 2
    model.add(Conv1D(8, 3))
    model.add(MaxPool1D())
    # # layer 3
    model.add(Conv1D(8, 3))
    model.add(MaxPool1D())

    model.add(Flatten())
    # model.add(Dropout(0.1))

    model.add(Dense(1))
    print(model.summary())
    plot_model(model, to_file='{}-{}feature.png'.format(model_input, number_feature), show_shapes=True, show_layer_names=True)

    model.compile(optimizer='Adam', loss='mae')
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

    plt.figure(global_stock)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig("Loss-{}-{}.png".format(filepath, global_stock))

    return model


def plot_predicted_result(y_cache, y_pred_all, stock, axes, i, mean, std):
    # plt.figure(figsize=(18, 6))

    y_pred = np.array(y_pred_all).mean(axis=0)
    print("y_pred")
    print(y_pred)
    axes[i].plot(y_cache * (std - mean) + mean)
    axes[i].plot(y_pred * (std - mean) + mean)
    axes[i].legend(["{} actual".format(stock), "{} forecast".format(stock)])
    # plt.title(stock)
    # plt.savefig(stock + "-predict-" + filepath + ".png")


def create_predicted_results_plot(test_datasets, y_pred, stocks, means, stds):
    fig, axs = plt.subplots(len(stocks), sharex=True, figsize=(16, 16))
    fig.suptitle('Sharing both axes')
    i = 0
    for stock in stocks:
        plot_predicted_result(test_datasets[stock], y_pred[stock], stock, axs,
                              i, means[stock], stds[stock])
        i += 1
    for ax in axs:
        ax.label_outer()
    plt.savefig("predict-" + filepath + ".png")



    # plt.figure(figsize=(18, 6))

    # plt.title(stock)
    # plt.savefig(stock + "-predict-" + filepath + ".png")


def predict(model, test_dataset: np.ndarray):
    x_test, y = next(test_dataset.as_numpy_iterator())
    print(x_test.shape)
    y_pred = model.predict(x_test)
    print(y_pred.shape)
    print(y.shape)


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
    return metric_results, test, test_pred


def create_bar_plot(results):
    labels = results.keys()
    print(np.array(list(results.values())))
    values = np.array(list(results.values()))
    print(values.shape)
    summary_means = values[:, 0]
    print("summary_means")
    print(str(summary_means))
    base_means = values[:, 1]
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
    # ax.set_yticks(np.arange(0, max([*summary_means, *base_means])))
    # ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.savefig("results-" + filepath + ".png")


train_size, val_size = 0.6, 0.2
features = list(range(0, 4)) #+ list(range(40, 70))# +[6,10,14,17,19,31,37,54,61,65]
# features = range(0, 1) #INPUT
print(features)
number_feature = len(features)
print(number_feature)
batch_size = 128
input_sequence_length = 60
forecast_horizon = 1
epochs = 200
add_all_datasets_data = True
model_input="CNNpred" #CNNpred, CNN-LSTM
filter=8
units=64
iter=5

filepath = "version2-{}-{}-{}-{}-{}-{}-{}-{}-{}".format(
    model_input,
    input_sequence_length,
    number_feature,
    epochs,
    batch_size,
    add_all_datasets_data,
    filter,
    units,
    iter
)

global_stock= ""

data_files_path = "LSTM/Datasets"
data_files_names = listdir(data_files_path)
stocks = []
train_datasets = {}
val_datasets = {}
test_datasets = {}
means = {}
stds = {}

for file in data_files_names:
    data = pd.read_csv("LSTM/Datasets/{}".format(file))
    stock = data['Name'][0]
    stocks.append(stock)
    del data["Name"]
    if number_feature > 1:
        data.insert(2, "Day", data['Date'].apply(lambda dt_str: datetime.strptime(dt_str, '%Y-%m-%d').weekday()))
    print(data.head())
    del data['Date']
    print(data.values.shape)
    data_array = data.values[:, features]
    print(data_array.shape)
    data_array = data_array[200:]
    data_array=np.nan_to_num(data_array)
    print(data_array.shape)
    print(data_array[:, 0])
    print("data_array[forecast_horizon:].shape")
    print(data_array[forecast_horizon:].shape)
    train_array, val_array, test_array, means[stock], stds[stock] = preprocess(data_array[forecast_horizon:], train_size, val_size, stock)

    # class_array = (data_array[forecast_horizon:] / data_array[:-forecast_horizon]).astype(int)
    # print("class_array.shape")
    # print(class_array.shape)
    # train_array_target, val_array_target, test_array_target = preprocess(class_array, train_size, val_size, stock)
    train_array_target, val_array_target, test_array_target = train_array, val_array, test_array

    train_dataset, val_dataset = (
        create_tf_dataset(data_array, target_array, input_sequence_length, forecast_horizon, batch_size)
        for (data_array, target_array) in [(train_array, train_array_target), (val_array, val_array_target)]
    )
    train_datasets[stock] = train_dataset
    val_datasets[stock] = val_dataset

    test_dataset = create_tf_dataset(
        test_array,
        test_array_target,
        input_sequence_length,
        forecast_horizon,
        batch_size=test_array.shape[0]
    )
    test_datasets[stock] = test_dataset

# results = {}
# y = {}
# y_pred = {}
# print(stocks)
# if add_all_datasets_data:
#
#     x = tf.data.Dataset.from_tensor_slices(train_datasets.values())
#     concat_ds = x.interleave(lambda x: x, cycle_length=1, num_parallel_calls=tf.data.AUTOTUNE)
#
#     x = tf.data.Dataset.from_tensor_slices(val_datasets.values())
#     concat_ds1 = x.interleave(lambda x: x, cycle_length=1, num_parallel_calls=tf.data.AUTOTUNE)
#
#     for m in range(0, iter):
#         if model_input=="CNN-LSTM":
#             model = CNN_LSTM_model(
#                 concat_ds,
#                 concat_ds1,
#                 input_sequence_length,
#                 number_feature,
#                 epochs
#             )
#
#
#         elif model_input == "CNNpred":
#             model = CNNpred_model(
#                 concat_ds,
#                 concat_ds1,
#                 input_sequence_length,
#                 number_feature,
#                 epochs
#             )
#
#         else:
#             model = FCN_LSTM_model(
#                 concat_ds,
#                 concat_ds1,
#                 input_sequence_length,
#                 number_feature,
#                 epochs
#             )
#
#
#         for stock in stocks:
#             r, y[stock], pred = predict(model, test_datasets[stock])
#             if stock in results.keys():
#                 list0 = list(results[stock])
#                 list0.append(r)
#                 results[stock] = list0
#                 list1 = list(y_pred[stock])
#                 list1.append(pred)
#                 y_pred[stock] = list1
#             else:
#                 results[stock] = [r]
#                 y_pred[stock] = [pred]
#
#     create_predicted_results_plot(y, y_pred, stocks, means, stds)
#
# else:
#
#
#     for stock in stocks:
#         global_stock = stock
#         for m in range(0, iter):
#             if model_input == "CNN-LSTM":
#                 model = CNN_LSTM_model(
#                     train_datasets[stock],
#                     val_datasets[stock],
#                     input_sequence_length,
#                     number_feature,
#                     epochs
#                 )
#
#             elif model_input=="CNNpred":
#                 model = CNNpred_model(
#                     train_datasets[stock],
#                     val_datasets[stock],
#                     input_sequence_length,
#                     number_feature,
#                     epochs
#                 )
#
#             else:
#                 model = FCN_LSTM_model(
#                     train_datasets[stock],
#                     val_datasets[stock],
#                     input_sequence_length,
#                     number_feature,
#                     epochs
#                 )
#             r, y[stock], pred = predict(model, test_datasets[stock])
#             if stock in results.keys():
#                 list0 = list(results[stock])
#                 list0.append(r)
#                 results[stock] = list0
#                 list1 = list(y_pred[stock])
#                 list1.append(pred)
#                 y_pred[stock] = list1
#             else:
#                 results[stock] = [r]
#                 y_pred[stock] = [pred]
#     create_predicted_results_plot(y, y_pred, stocks, means, stds)
#
# print("results")
# print(results)
# results = {k: np.array(v).mean(axis=0) for k,v in results.items()}
# create_bar_plot(results)
# pd.DataFrame.from_dict(results, orient='index', columns=["MAE", "RMSE"]).to_csv("results-" + filepath + ".csv")
