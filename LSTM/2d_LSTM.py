from os import listdir

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.utils import timeseries_dataset_from_array
from keras.utils.vis_utils import plot_model


def preprocess(data_array: np.ndarray, train_size: float, val_size: float):
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
    train_array = data_array[:num_train]
    print(train_array.shape)
    mean  = train_array.mean(axis=0),
    std =  train_array.std(axis=0)

    train_array = (train_array - mean) / std
    print("train_array")
    print(train_array)
    val_array = (data_array[num_train: (num_train + num_val)] - mean) / std
    test_array = (data_array[(num_train + num_val):] - mean) / std

    plt.figure(figsize=(18, 6))
    plt.plot(train_array[:, 0])
    plt.plot(val_array[:, 0])
    plt.plot(test_array[:, 0])
    plt.legend(["train", "val", "test"])
    plt.show()

    return train_array, val_array, test_array


def create_tf_dataset(
        data_array: np.ndarray,
        input_sequence_length: int,
        forecast_horizon: int,
        batch_size: int = 128,
        multi_horizon=True
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
        data_array[:-forecast_horizon],
        None,
        sequence_length=input_sequence_length,
        shuffle=False,
        batch_size=batch_size,
    )

    target_offset = (
        input_sequence_length
        if multi_horizon
        else input_sequence_length + forecast_horizon - 1
    )
    target_seq_length = forecast_horizon if multi_horizon else 1
    targets = timeseries_dataset_from_array(
        (data_array[target_offset:])[:, 0],
        None,
        sequence_length=target_seq_length,
        shuffle=False,
        batch_size=batch_size,
    )

    dataset = tf.data.Dataset.zip((inputs, targets))
    # for batch in dataset:
    #     inputs, targets = batch
    #     print(inputs.shape)
    #     print(targets.shape)

    print(type(dataset))
    print(dataset._variant_tensor)
    return dataset.prefetch(16).cache()


def LSTM_model(
        train_dataset: np.ndarray,
        val_dataset: np.ndarray,
        input_sequence_length: int,
        number_feature: int,
        epochs: int
):
    model = Sequential()
    model.add(LSTM(64, input_shape=(input_sequence_length, number_feature)))
    # model.add(LSTM(64, input_shape=(input_sequence_length, number_feature), return_sequences=True))
    # model.add(LSTM(64, return_sequences=True))
    # model.add(LSTM(32))
    model.add(Dense(1))
    print(model.summary())
    plot_model(model, to_file='model_plot_LSTM.png', show_shapes=True, show_layer_names=True)

    model.compile(loss='mae', optimizer='adam')
    model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

    return model

def predict(test_dataset: np.ndarray):
    x_test, y = next(test_dataset.as_numpy_iterator())
    print(x_test.shape)
    y_pred = model.predict(x_test)
    print(y_pred.shape)
    plt.figure(figsize=(18, 6))
    plt.plot(y[:, 0])
    plt.plot(y_pred[:, 0])
    plt.legend(["actual", "forecast"])
    plt.show()
    plt.savefig(filepath + ".png")

    naive_mse, model_mse = (
        np.square(x_test[:, -1, 0] - y[:, 0]).mean(),
        np.square(y_pred[:, 0] - y[:, 0]).mean(),
    )

    print(f"naive MAE: {naive_mse}, model MAE: {model_mse}")
    #DJI sin factor-> naive MAE: 0.0021673174502306954, model MAE: 0.060806224362133544
    #DJI range(1,10)-> naive MAE: 0.0021673174502306954, model MAE: 0.16460517128944463
    #DJI range(1,5)-> naive MAE: naive MAE: 0.0021673174502306954, model MAE: 0.2585225251668954
    #DJI range(1,5), DJI data x5-> naive MAE: 0.16946691639290737, model MAE: 0.10615230371562731
    #DJI range(1,2), all stocks-> naive MAE: 0.16946691639290737, model MAE: 0.08557734903886727
    #DJI range(1,5), all stocks-> naive MAE: 0.16946691639290737, model MAE: 0.09112090396689344



train_size, val_size = 0.6, 0.2
# features = list(range(1,58))+list(range(59,83))
features = range(1, 2)
print(features)
number_feature = len(features)
print(number_feature)
batch_size = 128
input_sequence_length = 60
forecast_horizon = 1
multi_horizon = False
epochs = 200

filepath = "predict-{}-{}-{}-{}".format(
            input_sequence_length,
            number_feature,
            epochs,
            batch_size
        )

data_files_path= "LSTM/Datasets"
data_files_names = listdir(data_files_path)
stocks=[]
train_datasets=[]
val_datasets=[]
test_dataset =[]


for file in data_files_names:
    data = pd.read_csv("LSTM/Datasets/{}".format(file))
    stock=data['Name'][0]
    stocks.append(stock)
    del data["Name"]
    del data["Date"]
    print(data.values.shape)
    data_array = data.values[:, features]
    print(data_array.shape)
    data_array = data_array[200:]
    print(data_array.shape)
    print(data_array[:, 0])
    train_array, val_array, test_array = preprocess(data_array, train_size, val_size)

    print(f"train set size: {train_array.shape}")
    print(f"validation set size: {val_array.shape}")
    print(f"test set size: {test_array.shape}")

    train_dataset, val_dataset = (
        create_tf_dataset(data_array, input_sequence_length, forecast_horizon, batch_size)
        for data_array in [train_array, val_array]
    )
    train_datasets.append(train_dataset)
    val_datasets.append(val_dataset)
    if stock == "DJI":
        test_dataset = create_tf_dataset(
            test_array,
            input_sequence_length,
            forecast_horizon,
            batch_size=test_array.shape[0],
            multi_horizon=multi_horizon,
        )

x = tf.data.Dataset.from_tensor_slices(train_datasets)
concat_ds = x.interleave(lambda x: x, cycle_length=1, num_parallel_calls=tf.data.AUTOTUNE)

x = tf.data.Dataset.from_tensor_slices(train_datasets)
concat_ds1 = x.interleave(lambda x: x, cycle_length=1, num_parallel_calls=tf.data.AUTOTUNE)

for b in concat_ds:
    i, t =b
    print("input")
    print(i.shape)
    print("target")
    print(t.shape)

print("--------------")
for b in concat_ds1:
    i, t =b
    print("input")
    print(i.shape)
    print("target")
    print(t.shape)

model = LSTM_model(
    concat_ds,
    concat_ds1,
    input_sequence_length,
    number_feature,
    epochs
)

results = predict(test_dataset)

