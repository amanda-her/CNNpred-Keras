import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from os.path import join
from sklearn.metrics import accuracy_score as accuracy, f1_score, mean_absolute_error as mae
import os
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling1D, Conv1D, Dense, RepeatVector, LSTM
from pathlib2 import Path
from keras import backend as K, callbacks


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        # print("true" + str(precision_pos))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        # print("precision" + str(precision_pos))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        # print("true " + str(true_positives.numpy()))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        # print("predicted " + str(predicted_positives.numpy()))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision_pos = precision(y_true, y_pred)
    # print("precision_POSITIVE "+str(precision_pos.numpy()))
    recall_pos = recall(y_true, y_pred)
    # print("recall_POSITIVE " + str(recall_pos.numpy()))
    precision_neg = precision((K.ones_like(y_true) - y_true), (K.ones_like(y_pred) - K.clip(y_pred, 0, 1)))
    # print("precision_NEGATIVE " + str(precision_pos.numpy()))
    recall_neg = recall((K.ones_like(y_true) - y_true), (K.ones_like(y_pred) - K.clip(y_pred, 0, 1)))
    # print("recall_NEGATIVE " + str(precision_pos.numpy()))
    f_posit = 2 * ((precision_pos * recall_pos) / (precision_pos + recall_pos + K.epsilon()))
    # print("f_POSITIVE " + str(f_posit.numpy()))
    f_neg = 2 * ((precision_neg * recall_neg) / (precision_neg + recall_neg + K.epsilon()))
    # print("f_NEGATIVE " + str(f_neg.numpy()))

    return (f_posit + f_neg) / 2


def load_data(file_fir):
    try:
        df_raw = pd.read_csv(file_fir, index_col='Date')  # parse_dates=['Date'])
    except IOError:
        print("IO ERROR")
    return df_raw


def costruct_data_warehouse(ROOT_PATH, file_names):
    global number_of_stocks
    global samples_in_each_stock
    global number_feature
    global order_stocks
    data_warehouse = {}

    for stock_file_name in file_names:

        print("STOCK " + stock_file_name)
        file_dir = os.path.join(ROOT_PATH, stock_file_name)
        ## Loading Data
        try:
            df_raw = load_data(file_dir)
        except ValueError:
            print("Couldn't Read {} file".format(file_dir))

        number_of_stocks += 1

        data = df_raw
        df_name = data['Name'][0]
        order_stocks.append(df_name)
        del data['Name']

        target = (data['Close'][predict_day:] / data['Close'][:-predict_day].values).astype(int)
        print("target " + str(target))
        print("target.shape " + str(target.shape))
        data = data[:-predict_day]
        target.index = data.index
        print("target.index " + str(target.index))
        # Becasue of using 200 days Moving Average as one of the features
        data = data[200:]
        data = data.fillna(0)
        data['target'] = target
        target = data['target']
        # data['Date'] = data['Date'].apply(lambda x: x.weekday())
        del data['target']

        print("data.shape " + str(data.shape))
        number_feature = data.shape[1]
        samples_in_each_stock = data.shape[0]

        train_data = data[data.index < '2016-04-21']
        print("train_data.shape " + str(train_data.shape))
        print("train_data " + str(train_data.head()))
        train_data1 = scale(train_data)
        # print("train_data1 " + str(train_data1[0:5, 0]))
        # print train_data.shape
        train_target1 = target[target.index < '2016-04-21']
        print("train_target1.shape " + str(train_target1.shape))
        print("train_target1 " + str(train_target1.head()))
        train_data = train_data1[:int(0.75 * train_data1.shape[0])]
        print("train_data.shape " + str(train_data.shape))
        # print("train_data " + str(train_data[0:5, 0]))
        train_target = train_target1[:int(0.75 * train_target1.shape[0])]
        print("train_target.shape " + str(train_target.shape))
        print("train_target " + str(train_target.head()))

        valid_data = scale(train_data1[int(0.75 * train_data1.shape[0]) - seq_len:])
        print("valid_data.shape " + str(valid_data.shape))
        # print("valid_data " + str(valid_data[0:5]))
        valid_target = train_target1[int(0.75 * train_target1.shape[0]) - seq_len:]
        print("valid_target.shape " + str(valid_target.shape))
        # print("valid_target " + str(valid_target[0:5]))

        data = pd.DataFrame(scale(data.values), columns=data.columns)
        data.index = target.index
        test_data = data[data.index >= '2016-04-21']
        print("test_data.shape " + str(test_data.shape))
        print("test_data " + str(test_data.head()))
        test_target = target[target.index >= '2016-04-21']
        print("test_target.shape " + str(test_target.shape))
        print("test_target " + str(test_target.head()))

        data_warehouse[df_name] = [train_data, train_target, np.array(test_data), np.array(test_target), valid_data,
                                   valid_target, data['Close'][predict_day:]]

    print(str(data_warehouse.items()))
    print(str(data_warehouse.values()))
    return data_warehouse


def cnn_data_sequence_separately(tottal_data, tottal_target, data, target, seque_len):
    print("cnn_data_sequence_separately")
    for index in range(data.shape[0] - seque_len + 1):
        tottal_data.append(data[index: index + seque_len])
        tottal_target.append(target[index + seque_len - 1])

    return tottal_data, tottal_target


def cnn_data_sequence(data_warehouse, seq_len):
    tottal_train_data = []
    tottal_train_target = []
    tottal_valid_data = []
    tottal_valid_target = []
    tottal_test_data = []
    tottal_test_target = []

    for key, value in data_warehouse.items():
        tottal_train_data, tottal_train_target = cnn_data_sequence_separately(tottal_train_data, tottal_train_target,
                                                                              value[0], value[1], seq_len)
        tottal_test_data, tottal_test_target = cnn_data_sequence_separately(tottal_test_data, tottal_test_target,
                                                                            value[2], value[3], seq_len)
        tottal_valid_data, tottal_valid_target = cnn_data_sequence_separately(tottal_valid_data, tottal_valid_target,
                                                                              value[4], value[5], seq_len)

    tottal_train_data = np.array(tottal_train_data)
    tottal_train_target = np.array(tottal_train_target)
    tottal_test_data = np.array(tottal_test_data)
    tottal_test_target = np.array(tottal_test_target)
    tottal_valid_data = np.array(tottal_valid_data)
    tottal_valid_target = np.array(tottal_valid_target)

    tottal_train_data = tottal_train_data.reshape(tottal_train_data.shape[0], tottal_train_data.shape[1],
                                                  tottal_train_data.shape[2], 1)
    tottal_test_data = tottal_test_data.reshape(tottal_test_data.shape[0], tottal_test_data.shape[1],
                                                tottal_test_data.shape[2], 1)
    tottal_valid_data = tottal_valid_data.reshape(tottal_valid_data.shape[0], tottal_valid_data.shape[1],
                                                  tottal_valid_data.shape[2], 1)
    print("tottal_train_data.shape " + str(tottal_train_data.shape))
    print("tottal_test_data.shape " + str(tottal_test_data.shape))
    print("tottal_valid_data.shape " + str(tottal_valid_data.shape))

    return tottal_train_data, tottal_train_target, tottal_test_data, tottal_test_target, tottal_valid_data, tottal_valid_target


def sklearn_acc(model, test_data, test_target):
    overall_results = model.predict(test_data)
    test_pred = (overall_results > 0.5).astype(int)
    a = test_target
    b = test_pred
    true_positives = len(set(np.where(a == 1)[0]).intersection(set(np.where(b == 1)[0])))
    true_negatives = len(set(np.where(a == 0)[0]).intersection(set(np.where(b == 0)[0])))
    false_positives = len(set(np.where(a == 0)[0]).intersection(set(np.where(b == 1)[0])))
    false_negatives = len(set(np.where(a == 1)[0]).intersection(set(np.where(b == 0)[0])))

    y = true_positives + false_positives
    precision = 1 if y == 0 else true_positives / y
    recall = true_positives / (true_positives + false_negatives)
    z = true_positives + false_positives
    f11 =  1 if z == 0 else 2 * precision * recall / z

    x = true_negatives + false_negatives
    precision_neg = 1 if x == 0 else true_negatives / x
    recall_neg = true_negatives / (true_negatives + false_positives)
    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg)

    accuracy1 = (true_positives + true_negatives) / (
                true_positives + true_negatives + false_positives + false_negatives)

    print("true_positives " + str(true_positives))
    print("true_negatives " + str(true_negatives))
    print("false_positives " + str(false_positives))
    print("false_negatives " + str(false_negatives))
    print("precision " + str(precision))
    print("recall " + str(recall))
    print("f1 " + str(f11))
    print("precision_neg " + str(precision_neg))
    print("recall_neg " + str(recall_neg))
    print("f1_neg " + str(f1_neg))
    print((f11 + f1_neg) / 2)
    print("accuracy " + str(accuracy1))

    acc_results = [mae(overall_results, test_target), accuracy(test_pred, test_target),
                   f1_score(test_pred, test_target, average='macro'),
                   f1(tf.convert_to_tensor(test_target.astype('float32')),
                      tf.convert_to_tensor(test_pred.astype('float32')))]

    print("overall_results " + str(len(overall_results)))
    print(str(overall_results))
    print("acc_results " + str(len(acc_results)))
    print(str(acc_results))

    return acc_results


def train(data_warehouse, i):
    print("TRAIN")
    seq_len = 60
    epochs = 200
    drop = 0.1

    global cnn_train_data, cnn_train_target, cnn_test_data, cnn_test_target, cnn_valid_data, cnn_valid_target

    if i == 1:
        print('sequencing ...')
        cnn_train_data, cnn_train_target, cnn_test_data, cnn_test_target, cnn_valid_data, cnn_valid_target = cnn_data_sequence(
            data_warehouse, seq_len)

    my_file = Path(join(Base_dir,
                        '2D-models/best-{}-{}-{}-{}-{}.h5'.format(epochs, seq_len, number_filter, drop, i)))
    filepath = join(Base_dir, '2D-models/best-{}-{}-{}-{}-{}.h5'.format(epochs, seq_len, number_filter, drop, i))
    if my_file.is_file():
        print('loading model')

    else:

        # input_shape = (seq_len, number_feature)
        # input_layer = Input(input_shape)
        #
        # conv1 = Conv1D(filters=256, kernel_size=8, padding='same')(input_layer)
        # print("conv1 " + str(conv1.shape))
        # conv1 = MaxPooling1D()(conv1)
        # print("conv1 " + str(conv1.shape))
        #
        # lstm1 = LSTM(units=256)(conv1)
        # print("lstm1 " + str(lstm1.shape))
        # vector = RepeatVector(lstm1, 1)
        # print("vector " + str(vector.shape))
        # lstm2 = LSTM(units=256)(vector)
        # print("lstm2 " + str(lstm2.shape))
        #
        # output_layer = Dense(1, activation='softmax')(lstm2)
        # print("output_layer " + str(output_layer.shape))

        print(' fitting model to target')
        model = Sequential()
        #
        # layer 1
        model.add(
            Conv1D(filters=256, kernel_size=8, activation='relu', input_shape=(seq_len, number_feature)))
        print("layer 1 " + str(model.output_shape))
        # layer 2
        model.add(MaxPooling1D())
        print("layer 2 " + str(model.output_shape))

        # layer 3
        model.add(LSTM(units=256))
        print("layer 3 " + str(model.output_shape))
        model.add(RepeatVector(1))
        print("layer 3 " + str(model.output_shape))
        model.add(LSTM(units=256))
        print("layer 3 " + str(model.output_shape))

        # model.add(Flatten())
        # print("layer flatten " + str(model.output_shape))
        # model.add(Dropout(drop))
        # print("layer dropout " + str(model.output_shape))

        model.add(Dense(1, activation='relu'))
        print("layer dense " + str(model.output_shape))

        model.compile(optimizer='Adam', loss='mae', metrics=['acc', f1])

        best_model = callbacks.ModelCheckpoint(filepath, monitor='val_f1', verbose=0, save_best_only=True,
                                               save_weights_only=False, mode='max', period=1)

        model.fit(cnn_train_data, cnn_train_target, epochs=epochs, batch_size=128, verbose=1,
                  validation_data=(cnn_valid_data, cnn_valid_target), callbacks=[best_model])
    model = load_model(filepath, custom_objects={'f1': f1})

    return model, seq_len


def cnn_data_sequence_pre_train(data, target, seque_len):
    print("cnn_data_sequence_pre_train")
    new_data = []
    new_target = []
    for index in range(data.shape[0] - seque_len + 1):
        new_data.append(data[index: index + seque_len])
        new_target.append(target[index + seque_len - 1])

    new_data = np.array(new_data)
    new_target = np.array(new_target)

    new_data = new_data.reshape(new_data.shape[0], new_data.shape[1], new_data.shape[2], 1)

    print("new_data.shape " + str(new_data.shape))
    print("new_target.shape " + str(new_target.shape))
    return new_data, new_target


def prediction(data_warehouse, model, seque_len, order_stocks, cnn_results):
    print("PREDICTION")
    for name in order_stocks:
        print("NAME " + name)
        value = data_warehouse[name]
        # train_data, train_target = cnn_data_sequence_pre_train(value[0], value[1], seque_len)
        test_data, test_target = cnn_data_sequence_pre_train(value[2], value[3], seque_len)
        # valid_data, valid_target = cnn_data_sequence_pre_train(value[4], value[5], seque_len)

        cnn_results[name] = np.append(cnn_results[name], sklearn_acc(model, test_data, test_target))
        print("cnn_results")
        print(str(cnn_results))

    return cnn_results


def run_cnn_ann(data_warehouse, order_stocks):
    cnn_results = dict((stock, np.empty(0)) for stock in order_stocks)
    print(str(cnn_results))
    # dnn_results = []
    iterate_no = 40
    for i in range(1, iterate_no):
        K.clear_session()
        print("iterate_no")
        print(i)
        model, seq_len = train(data_warehouse, i)
        # cnn_results, dnn_results = prediction(data_warehouse, model, seq_len, order_stocks, cnn_results)
        cnn_results = prediction(data_warehouse, model, seq_len, order_stocks, cnn_results)

    for stock in cnn_results.keys():
        cnn_results1 = cnn_results[stock].reshape(iterate_no - 1, 4)
        cnn_results1 = pd.DataFrame(cnn_results1, columns=["mae", "accuracy", "f1", "f1_custom"])
        cnn_results1 = cnn_results1.append([cnn_results1.mean(), cnn_results1.max(), cnn_results1.std()],
                                           ignore_index=True)
        cnn_results1.to_csv(join(Base_dir, '2D-models/{}/new results-LSTM.csv'.format(stock)), index=False)


Base_dir = ''
TRAIN_ROOT_PATH = join(Base_dir, 'Dataset')
train_file_names = os.listdir(join(Base_dir, 'Dataset'))

# if moving average = 0 then we have no moving average
seq_len = 60
moving_average_day = 0
number_of_stocks = 0
number_feature = 0
samples_in_each_stock = 0
number_filter = [8, 8, 8]
predict_day = 1

cnn_train_data, cnn_train_target, cnn_test_data, cnn_test_target, cnn_valid_data, cnn_valid_target = ([] for i in
                                                                                                      range(6))

print('Loading train data ...')
order_stocks = []
data_warehouse = costruct_data_warehouse(TRAIN_ROOT_PATH, train_file_names)
# order_stocks = data_warehouse.keys()

print('number of stocks = '), number_of_stocks

run_cnn_ann(data_warehouse, order_stocks)
