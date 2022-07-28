from os import listdir
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import Conv2D, MaxPool2D, Conv1D, BatchNormalization, Activation, GlobalAveragePooling1D, Dense, \
    Dropout, Flatten, Input, MaxPooling1D, LSTM, MaxPool1D
from keras.metrics import Accuracy
from keras.models import Sequential, load_model, Model
from keras.optimizers import Adam
from pathlib2 import Path
from sklearn.metrics import accuracy_score as accuracy, f1_score, mean_absolute_error as mae, mean_squared_error as mse
from sklearn.preprocessing import OneHotEncoder, scale
from keras.utils.vis_utils import plot_model


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision_pos = precision(y_true, y_pred)
    recall_pos = recall(y_true, y_pred)
    precision_neg = precision((K.ones_like(y_true) - y_true), (K.ones_like(y_pred) - K.clip(y_pred, 0, 1)))
    recall_neg = recall((K.ones_like(y_true) - y_true), (K.ones_like(y_pred) - K.clip(y_pred, 0, 1)))
    f_posit = 2 * ((precision_pos * recall_pos) / (precision_pos + recall_pos + K.epsilon()))
    f_neg = 2 * ((precision_neg * recall_neg) / (precision_neg + recall_neg + K.epsilon()))

    return (f_posit + f_neg) / 2


def load_data(file_fir):
    df_raw = {}
    try:
        df_raw = pd.read_csv(file_fir, index_col='Date')
    except IOError:
        print("IO ERROR")
    return df_raw


def load_base_results():
    df_results = []
    try:
        df_results = pd.read_csv("2D-models/base-results.csv", index_col='stock')
    except IOError:
        print("IO ERROR")
    return df_results


def prepare_target(data, predict_day):
    if model_param == "FCN":
        target_values = []
        if nb_classes == 2:
            target_values = (data[predict_day:] / data[:-predict_day].values).astype(int).values
        elif nb_classes == 3:
            a = data
            b = np.array([a[2:], a[1:-1], a[:-2]])
            class_array = []
            for i in range(b.shape[1]):
                x = b[:, i]
                print("x1")
                print(x)
                x = x[::-1]
                print("x2")
                print(x)
                if x[0] > x[1] > x[2]:  # 3 2 1
                    class_array.append(1)  # =
                elif x[0] > x[1] and x[1] < x[0]:  # 3 2 3
                    class_array.append(2)  # sube
                elif x[0] > x[1] == x[2]:  # 3 2 2
                    class_array.append(1)  # =
                elif x[0] == x[1] and x[1] > x[2]:  # 3 3 2
                    class_array.append(0)  # baja
                elif x[0] == x[1] and x[1] < x[2]:  # 3 3 4
                    class_array.append(2)  # sube
                elif x[0] == x[1] and x[1] == x[2]:  # 3 3 3
                    class_array.append(1)  # =
                elif x[0] < x[1] and x[1] > x[2]:  # 1 2 1
                    class_array.append(0)  # baja
                elif x[0] < x[1] < x[2]:  # 1 2 3
                    class_array.append(1)  # =
                elif x[0] < x[1] == x[2]:  # 1 2 2
                    class_array.append(1)  # =
            print(len(class_array))
            target_values = np.array(np.insert(np.array(class_array), 0, 1))
            print(len(target_values))

        # transform the labels from integers to one hot vectors
        enc = OneHotEncoder(categories='auto')
        target_values_reshaped = target_values.reshape(-1, 1)
        enc.fit(target_values_reshaped)
        target = pd.DataFrame(enc.transform(target_values_reshaped).toarray())
    else:  # CNNpred or CNN-LSTM
        target = (data[predict_day:] / data[:-predict_day].values).astype(int)

    return target


def costruct_data_warehouse(ROOT_PATH, file_names):
    global number_feature
    predict_day = 1
    data_warehouse = {}

    for stock_file_name in file_names:

        file_dir = join(ROOT_PATH, stock_file_name)
        ## Loading Data
        try:
            df_raw = load_data(file_dir)
        except ValueError:
            print("Couldn't Read {} file".format(file_dir))

        data = df_raw
        df_name = data['Name'][0]
        order_stocks.append(df_name)
        del data['Name']

        target = prepare_target(data['Close'], predict_day)
        print("target")
        print(target.shape)
        data = data[:-predict_day]
        # Becasue of using 200 days Moving Average as one of the features
        data = data[200:]
        data = data.fillna(0)
        target = target[200:]
        target.index = data.index

        number_feature = data.shape[1]
        samples_in_each_stock = data.shape[0]
        print("Stock: {}, number features: {}, samples in stock: {}.".format(df_name, number_feature,
                                                                             samples_in_each_stock))

        train_valid_data = data[data.index < '2016-04-21']
        train_valid_data_scaled = scale(train_valid_data) if scale_param else train_valid_data
        train_valid_target = target[target.index < '2016-04-21']
        train_data = train_valid_data_scaled[:int(0.75 * train_valid_data_scaled.shape[0])]
        train_target = train_valid_target[:int(0.75 * train_valid_target.shape[0])]
        valid_data = train_valid_data_scaled[int(0.75 * train_valid_data_scaled.shape[0]) - seq_len:]
        valid_data_scaled = scale(valid_data) if scale_param else valid_data
        valid_target = train_valid_target[int(0.75 * train_valid_target.shape[0]) - seq_len:]

        data = pd.DataFrame((scale(data.values) if scale_param else data.values), columns=data.columns)
        data.index = target.index
        test_data = data[data.index >= '2016-04-21']
        test_target = target[target.index >= '2016-04-21']

        data_warehouse[df_name] = [train_data, train_target, np.array(test_data), np.array(test_target),
                                   valid_data_scaled, valid_target]

    return data_warehouse


def cnn_data_sequence_separately(tottal_data, tottal_target, data, target, seque_len):
    for index in range(data.shape[0] - seque_len + 1):
        tottal_data.append(data[index: index + seque_len])
        tottal_target.append(target[index + seque_len - 1])

    return tottal_data, tottal_target


def cnn_data_sequence(data_warehouse):
    tottal_train_data = []
    tottal_train_target = []
    tottal_valid_data = []
    tottal_valid_target = []
    tottal_test_data = []
    tottal_test_target = []

    for key, value in data_warehouse.items():
        tottal_train_data, tottal_train_target = cnn_data_sequence_separately(tottal_train_data, tottal_train_target,
                                                                              value[0], np.array(value[1]), seq_len)
        tottal_test_data, tottal_test_target = cnn_data_sequence_separately(tottal_test_data, tottal_test_target,
                                                                            value[2], value[3], seq_len)
        tottal_valid_data, tottal_valid_target = cnn_data_sequence_separately(tottal_valid_data, tottal_valid_target,
                                                                              value[4], np.array(value[5]), seq_len)

    tottal_train_data = np.array(tottal_train_data)
    tottal_train_target = np.array(tottal_train_target)
    tottal_test_data = np.array(tottal_test_data)
    tottal_test_target = np.array(tottal_test_target)
    tottal_valid_data = np.array(tottal_valid_data)
    tottal_valid_target = np.array(tottal_valid_target)

    if model_param == "CNNpred":
        tottal_train_data = tottal_train_data.reshape(tottal_train_data.shape[0], tottal_train_data.shape[1],
                                                      tottal_train_data.shape[2], 1)
        tottal_test_data = tottal_test_data.reshape(tottal_test_data.shape[0], tottal_test_data.shape[1],
                                                    tottal_test_data.shape[2], 1)
        tottal_valid_data = tottal_valid_data.reshape(tottal_valid_data.shape[0], tottal_valid_data.shape[1],
                                                      tottal_valid_data.shape[2], 1)
    print("tottal_train_data.shape " + str(tottal_train_data.shape))
    print("tottal_test_data.shape " + str(tottal_test_data.shape))
    print("tottal_valid_data.shape " + str(tottal_valid_data.shape))
    print("tottal_train_target.shape " + str(tottal_train_target.shape))
    print("tottal_test_target.shape " + str(tottal_test_target.shape))
    print("tottal_valid_target.shape " + str(tottal_valid_target.shape))

    return tottal_train_data, tottal_train_target, tottal_test_data, tottal_test_target, tottal_valid_data, tottal_valid_target


def sklearn_acc(model, test_data, test_target):
    overall_results = model.predict(test_data)
    print("overall_results")
    print(overall_results)
    if model_param == "FCN":
        test_pred = np.argmax(overall_results, axis=1)
        test_target1 = np.argmax(test_target, axis=1)
    else:  # CNNpred or CNN-LSTM
        test_pred = (overall_results > 0.5).astype(int)
        test_target1 = test_target
    print("test_pred")
    print(test_pred)
    print("test_target")
    print(test_target)

    acc_results = [mae(overall_results, test_target), accuracy(test_pred, test_target1),
                   f1_score(test_pred, test_target1, average='macro'), mse(overall_results, test_target)]

    return acc_results


def compile():
    model = Sequential()

    if model_param == "CNNpred":
        # layer 1
        model.add(
            Conv2D(number_filter[0], (1, number_feature), activation='relu', input_shape=(seq_len, number_feature, 1))
        )
        # layer 2
        model.add(Conv2D(number_filter[1], (3, 1), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 1)))
        # layer 3
        model.add(Conv2D(number_filter[2], (3, 1), activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 1)))

        model.add(Flatten())
        model.add(Dropout(drop))

        model.add(Dense(1, activation='sigmoid'))
        print(model.summary())
        plot_model(model, to_file='model_plot_CNNpred.png', show_shapes=True, show_layer_names=True)

        model.compile(optimizer='Adam', loss='mae', metrics=metrics)

    elif model_param == "CNNpred-1D":
        # layer 1
        model.add(
            Conv1D(number_filter[0], 1, activation='relu', input_shape=(seq_len, number_feature))
        )
        # layer 2
        model.add(Conv1D(number_filter[1], 3, activation='relu'))
        model.add(MaxPool1D())
        # layer 3
        model.add(Conv1D(number_filter[2], 3, activation='relu'))
        model.add(MaxPool1D())

        model.add(Flatten())
        model.add(Dropout(drop))

        model.add(Dense(1, activation='sigmoid'))
        print(model.summary())
        plot_model(model, to_file='model_plot_CNNpred-1D.png', show_shapes=True, show_layer_names=True)

        model.compile(optimizer='Adam', loss='mae', metrics=metrics)

    elif model_param == "FCN":
        input_shape = (seq_len, number_feature)
        input_layer = Input(input_shape)

        conv1 = Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
        conv1 = BatchNormalization()(conv1)
        conv1 = Activation(activation='relu')(conv1)

        conv2 = Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = Activation('relu')(conv2)

        conv3 = Conv1D(128, kernel_size=3, padding='same')(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = Activation('relu')(conv3)

        gap_layer = GlobalAveragePooling1D()(conv3)

        output_layer = Dense(nb_classes, activation='softmax')(gap_layer)

        model = Model(inputs=input_layer, outputs=output_layer)
        print(model.summary())
        plot_model(model, to_file='model_plot_FCN.png', show_shapes=True, show_layer_names=True)

        model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=[Accuracy()])

    elif model_param == "CNN-LSTM":
        # layer 1
        model.add(
            Conv1D(filters=256, kernel_size=1, activation='tanh', input_shape=(seq_len, number_feature),
                   padding="same"))
        print("layer 1 " + str(model.output_shape))
        # model.add(
        #     Conv1D(filters=256, kernel_size=1, activation='relu', input_shape=(seq_len, number_feature)))
        # print("layer 1 " + str(model.output_shape))
        # layer 2
        model.add(MaxPooling1D(pool_size=1, padding="same"))
        print("layer 2 " + str(model.output_shape))

        # layer 3
        model.add(LSTM(units=512, recurrent_activation="tanh"))
        print("layer 3 " + str(model.output_shape))
        # model.add(RepeatVector(1))
        # print("layer 3 " + str(model.output_shape))
        # model.add(LSTM(units=256))
        # print("layer 3 " + str(model.output_shape))

        model.add(Dense(1))
        print("layer dense " + str(model.output_shape))
        print(model.summary())
        plot_model(model, to_file='model_plot_CNN-LSTM.png', show_shapes=True, show_layer_names=True)

        model.compile(optimizer='Adam', loss='mae', metrics=[Accuracy()])

    return model


def train(compiled_model, data_warehouse, i, j):
    global cnn_train_data, cnn_train_target, cnn_test_data, cnn_test_target, cnn_valid_data, cnn_valid_target
    filepath = join(
        Base_dir,
        '2D-models/best-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}.h5'.format(
            model_param + str(nb_classes) if model_param == "FCN" else model_param,
            seq_len,
            number_filter,
            epochs,
            batch_size,
            compiling_iter,
            fitting_iter,
            drop,
            scale_param,
            print_metrics(),
            activation,
            optimizer,
            loss,
            i,
            j
        )
    )
    my_file = Path(filepath)
    custom_objects = {}

    if i == 1:
        print('sequencing ...')
        cnn_train_data, cnn_train_target, cnn_test_data, cnn_test_target, cnn_valid_data, cnn_valid_target = \
            cnn_data_sequence(data_warehouse)

    if my_file.is_file():
        print('loading model')
        if "CNNpred" in model_param:
            custom_objects = {'f1': f1}
    else:
        print(' fitting model to target')
        if "CNNpred" in model_param:
            best_model = ModelCheckpoint(filepath, monitor='val_f1', verbose=0, save_best_only=True,
                                         save_weights_only=False, mode='max', period=1)
            compiled_model.fit(cnn_train_data, cnn_train_target, epochs=epochs, batch_size=batch_size, verbose=1,
                               validation_data=(cnn_valid_data, cnn_valid_target), callbacks=[best_model])
            custom_objects = {'f1': f1}

        elif model_param == "FCN":
            reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)
            model_checkpoint = ModelCheckpoint(filepath, monitor='loss', save_best_only=True)
            mini_batch_size = int(min(cnn_train_target.shape[0] / 10, batch_size))
            compiled_model.fit(cnn_train_data, cnn_train_target, batch_size=mini_batch_size, epochs=epochs,
                               verbose=1, validation_data=(cnn_valid_data, cnn_valid_target),
                               callbacks=[reduce_lr, model_checkpoint])

        elif model_param == "CNN-LSTM":
            best_model = ModelCheckpoint(filepath, monitor='loss', verbose=0, save_best_only=True,
                                         save_weights_only=False, mode='max', period=1)
            compiled_model.fit(cnn_train_data, cnn_train_target, epochs=epochs, batch_size=batch_size, verbose=1,
                               validation_data=(cnn_valid_data, cnn_valid_target), callbacks=[best_model])

    model = load_model(filepath, custom_objects=custom_objects)

    return model


def cnn_data_sequence_pre_train(data, target):
    new_data = []
    new_target = []
    for index in range(data.shape[0] - seq_len + 1):
        new_data.append(data[index: index + seq_len])
        new_target.append(target[index + seq_len - 1])

    new_data = np.array(new_data)
    new_target = np.array(new_target)

    new_data = new_data.reshape(new_data.shape[0], new_data.shape[1], new_data.shape[2], 1)

    return new_data, new_target


def prediction(data_warehouse, model, cnn_results):
    for name in order_stocks:
        value = data_warehouse[name]
        # train_data, train_target = cnn_data_sequence_pre_train(value[0], value[1], seque_len)
        test_data, test_target = cnn_data_sequence_pre_train(value[2], value[3])
        # valid_data, valid_target = cnn_data_sequence_pre_train(value[4], value[5], seque_len)

        cnn_results[name] = np.append(cnn_results[name], sklearn_acc(model, test_data, test_target))
        # cnn_results[name] = np.append(cnn_results[name], sklearn_acc(model, train_data, train_target))
        # cnn_results[name] = np.append(cnn_results[name], sklearn_acc(model, valid_data, valid_target))

    return cnn_results


def run_cnn_ann(data_warehouse, order_stocks):
    cnn_results = dict((stock, np.empty(0)) for stock in order_stocks)
    summary_results = dict((stock, np.empty(0)) for stock in order_stocks)
    columns = ["mae", "accuracy", "f1", "mse"]

    for i in range(1, compiling_iter):
        compiled_model = compile()
        for j in range(1, fitting_iter):
            model = train(compiled_model, data_warehouse, i, j)
            cnn_results = prediction(data_warehouse, model, cnn_results)
            K.clear_session()

    for stock in cnn_results.keys():
        cnn_results1 = cnn_results[stock].reshape((compiling_iter - 1) * (fitting_iter - 1), 4)
        cnn_results1 = pd.DataFrame(cnn_results1, columns=columns)
        cnn_results1 = cnn_results1.append([cnn_results1.mean(), cnn_results1.max(), cnn_results1.std()],
                                           ignore_index=True)
        cnn_results1.to_csv(join(
            Base_dir,
            '2D-models/{}/results-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}.csv'.format(
                stock,
                model_param + str(nb_classes) if model_param == "FCN" else model_param,
                seq_len,
                number_filter,
                epochs,
                batch_size,
                compiling_iter,
                fitting_iter,
                drop,
                scale_param,
                activation,
                optimizer,
                loss,
                print_metrics()
            )
        ), index=False)
        summary_results[stock] = cnn_results1.mean()

    base_results = load_base_results()
    df_summary_results = pd.DataFrame.from_dict(summary_results).transpose()
    for c in columns:
        create_bar_plot(c, df_summary_results.loc[:, c], base_results.loc[:, c])


def create_bar_plot(column, summary_results, base_results):
    labels = order_stocks
    summary_means = summary_results
    base_means = base_results

    # https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, summary_means, width, label='Modified model')
    ax.bar(x + width / 2, base_means, width, label='Base model')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(column)
    ax.set_yticks(np.arange(0, max(summary_means.values.tolist() + base_means.values.tolist()) + 0.1, 0.1))
    # ax.set_title('Scores by group and gender')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(loc='lower right')
    fig.tight_layout()
    plt.savefig(join(
        Base_dir,
        '2D-models/Figures/{}/results-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}.png'.format(
            column,
            model_param + str(nb_classes) if model_param == "FCN" else model_param,
            seq_len,
            number_filter,
            epochs,
            batch_size,
            compiling_iter,
            fitting_iter,
            drop,
            scale_param,
            activation,
            optimizer,
            loss,
            print_metrics()
        )
    ))


def print_metrics():
    str_metrics = []
    for m in metrics:
        try:
            str_metrics.append(m.name)
        except:
            str_metrics.append(m.__name__)
    return str(str_metrics)


def run_model_with_params(
        seq_len_parametrized: int,
        number_filter_parametrized: list,
        epochs_parametrized: int,
        batch_size_parametrized: int,
        compiling_iter_parametrized: int,
        fitting_iter_parametrized: int,
        drop_parametrized: float,
        activation_parametrized: str,
        optimizer_parametrized: str,
        loss_parametrized: str,
        metrics_parametrized: list
):
    global seq_len, number_filter, epochs, batch_size, compiling_iter, fitting_iter, drop, \
        activation, optimizer, loss, metrics

    seq_len = seq_len_parametrized
    number_filter = number_filter_parametrized
    epochs = epochs_parametrized
    batch_size = batch_size_parametrized
    compiling_iter = compiling_iter_parametrized
    fitting_iter = fitting_iter_parametrized
    drop = drop_parametrized
    activation = activation_parametrized
    optimizer = optimizer_parametrized
    loss = loss_parametrized
    metrics = metrics_parametrized

    run_cnn_ann(data_warehouse, order_stocks)


Base_dir = ''
TRAIN_ROOT_PATH = join(Base_dir, 'Dataset')
train_file_names = listdir(TRAIN_ROOT_PATH)
number_feature = 0
seq_len = 0
number_filter = []
epochs = 0
batch_size = 0
compiling_iter = 0
fitting_iter = 0
drop = 0
scale_param = True
activation = ""
optimizer = ""
loss = ""
metrics = []
model_param = "CNNpred-1D"  # CNNpred, FCN, CNN-LSTM, CNNpred-1D
nb_classes = 3  # param needed when model_param == FCN

cnn_train_data, cnn_train_target, cnn_test_data, cnn_test_target, cnn_valid_data, cnn_valid_target = ([] for i in
                                                                                                      range(6))
print('Loading train data ...')
order_stocks = []
data_warehouse = costruct_data_warehouse(TRAIN_ROOT_PATH, train_file_names)
print('Number of stocks: {}'.format(len(order_stocks))),

run_model_with_params(
    seq_len_parametrized=60,
    number_filter_parametrized=[8, 8, 8],
    epochs_parametrized=200,
    batch_size_parametrized=128,
    compiling_iter_parametrized=6,
    fitting_iter_parametrized=2,
    drop_parametrized=0.1,
    activation_parametrized='sigmoid',
    optimizer_parametrized='Adam',
    loss_parametrized='mae',
    metrics_parametrized=[Accuracy(), f1]
)
