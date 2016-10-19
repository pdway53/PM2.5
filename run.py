# -*- coding: utf-8 -*-
# @Author: aaronlai
# @Date:   2016-10-15 23:19:23
# @Last Modified by:   aaronlai
# @Last Modified time: 2016-10-16 00:40:34

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # noqa
import matplotlib.pyplot as plt


def process_data(filename, skiprow=0):
    """
    Load and process data into a list of pandas DataFrame
    each element in the list = one id
    """
    df = pd.read_csv(filename, encoding='big5', header=None, skiprows=skiprow)
    # drop 測站
    df.drop(1, axis=1, inplace=True)
    print('Data Loaded, preview:')
    print(df.head())

    data = {}
    # group data by date
    for name, ddf in df.groupby(0):
        date = [s.zfill(2) for s in name.split('/')]
        month = date[1]

        # drop the date
        ddf.drop(0, axis=1, inplace=True)

        # set index as the measure
        ddf.set_index(2, drop=True, inplace=True)

        # set column as month-day-hour
        ddf.columns = ['-'.join(date[1:]+[str(i).zfill(2)]) for i in range(24)]

        # concatenate
        if month in data:
            data[month] = pd.concat([data[month], ddf], axis=1)
        else:
            data[month] = ddf

    # sort the columns by datetime
    for key in data.keys():
        data[key] = data[key][data[key].columns.sort_values()]

    print('\nShow data index:')
    print(data['01'].columns)

    return data


def generate_dateset(data, monthly_size=20):
    """sample and generate data from a list of pandas DataFrame"""
    data_id = 0
    data_X = []
    data_y = []

    for month in range(1, 13):
        start_points_num = len(data[str(month).zfill(2)].columns) - 9

        if monthly_size == 'all':
            indexes = list(range(start_points_num))
        else:
            # random sample from each month
            indexes = np.random.choice(start_points_num, monthly_size,
                                       replace=False)

        for ind, i in enumerate(indexes):
            dff = data[str(month).zfill(2)]

            X = dff[dff.columns[i:i + 9]].reset_index()

            # add id column
            X[0] = 'id_{}'.format(str(data_id))

            # change column order
            X = X[[X.columns[-1]] + X.columns[:-1].tolist()]
            X.columns = [i for i in range(len(X.columns))]

            PM25 = dff[dff.columns[i + 9]].loc['PM2.5']
            y = ['id_{}'.format(str(data_id)), PM25]

            data_X.append(X)
            data_y.append(y)

            data_id += 1

    return data_X, data_y


def make_data(train_X, train_y, k=None):
    """make data to be Matrix-like"""
    X = []
    y = []
    for i in range(len(train_y)):
        # only use PM2.5 as features
        x = train_X[i][train_X[i][1] == 'PM2.5']
        # replace NoRain as 0.0
        x = x.replace('NR', '0.0')[list(range(2, 11))]
        x = x.applymap(lambda x: float(x))
        X.append(x.values.ravel())
        y.append(float(train_y[i][1]))

    X = np.array(X)
    y = np.array(y)

    if k:
        X = X[:, -k:]

    # add bias
    X = np.concatenate((X, np.ones((X.shape[0], 1))), axis=1)

    return X, y


def LinReg_fit(X, y, X_test=None, y_test=None, lr=1e-7, batch=1, lamb=0,
               epoch=10000, print_every=100, lamb1=0, momentum=0):
    """train Linear Regression by adagrad"""
    # initialize
    W = np.random.randn(X.shape[1]) / X.shape[1] / X.shape[0]

    train_loss = []
    train_RMSE = []
    test_loss = []
    test_RMSE = []

    # batch size indicator
    b = 0
    # cache for adagrad
    G = np.zeros(W.shape)

    for i in range(epoch):
        inds = []
        last_step = 0

        for j in np.random.permutation(X.shape[0]):
            inds.append(j)
            b += 1

            # do the adagrad to update the parameter
            if b >= batch:
                diff = X[inds].dot(W) - y[inds]

                # calculate gradients
                w = np.array(W)
                w[w > 0] = 1
                w[w < 0] = -1
                grad_X = X[inds].T.dot(diff)
                grad_regulariz = lamb * W * batch / X.shape[0]
                grad_first_order_reg = lamb1 * w * batch / X.shape[0]
                grad = grad_X + grad_regulariz + grad_first_order_reg

                # calculate update step
                G += grad**2
                delta_W = (grad + momentum * last_step) / np.sqrt(G)
                W -= lr * delta_W

                # reset variables
                last_step = delta_W
                b = 0
                inds = []

        objective = (((X.dot(W) - y)**2).sum() + lamb * (W**2).sum()) / 2.0
        RMSE = cal_RMSE(X, W, y)

        if X_test is not None and y_test is not None:
            # losses
            loss_X = ((X_test.dot(W) - y_test)**2).sum() / 2.0
            loss_reg = lamb * (W**2).sum() / 2.0
            loss_first_reg = lamb1 * (abs(W).sum())

            obj_t = loss_X + loss_reg + loss_first_reg
            RMSE_t = cal_RMSE(X_test, W, y_test)

            test_loss.append(obj_t)
            test_RMSE.append(RMSE_t)

        # print out the progress
        if i % print_every == 0:
            if X_test is not None and y_test is not None:
                print('\tepoch: %d; obj: %.4f; RMSE: %.4f; RMSE_test: %.4f' %
                      (i, objective, RMSE, RMSE_t))
            else:
                print('\tepoch: %d; obj: %.4f; RMSE: %.4f' %
                      (i, objective, RMSE))

        train_loss.append(objective)
        train_RMSE.append(RMSE)

    print('final obj: %.4f' % train_loss[-1])

    return W, train_loss, train_RMSE, test_loss, test_RMSE


def cal_RMSE(X, W, y):
    """Calculate the RMSE"""
    return np.sqrt(((X.dot(W) - y) ** 2).sum() / len(y))


def train_valid_split(X, y):
    """Split the data as training and validation sets"""
    random_indexes = np.random.permutation(len(y))
    train_inds = random_indexes[:(0.75*len(y))]
    valid_inds = random_indexes[(0.75*len(y)):]
    return X[train_inds], y[train_inds], X[valid_inds], y[valid_inds]


def run_baseline(split=True):
    """baseline for linear regression"""
    data = process_data('train.csv', skiprow=1)
    # sample how many data a month, use int or 'all'
    train_X, train_y = generate_dateset(data, monthly_size='all')

    X, y = make_data(train_X, train_y, k=6)

    # split out validation set or not
    if split:
        X_train, y_train, X_valid, y_valid = train_valid_split(X, y)
    else:
        X_train, y_train = X, y
        X_valid = None
        y_valid = None

    # training
    print('\nStart training...')
    # no L1 or L2 regularization
    result = LinReg_fit(X_train, y_train, X_test=X_valid, y_test=y_valid,
                        lr=2e-2, batch=5, lamb=0, epoch=2000,
                        print_every=200)
    W, train_loss, train_RMSE, valid_loss, valid_RMSE = result

    # calculate final RMSE
    loss = ((X.dot(W) - y)**2).sum() / 2.0
    RMSE = cal_RMSE(X, W, y)
    RMSE_t = cal_RMSE(X_valid, W, y_valid)
    print('Loss: %.4f; RMSE: %.4f; RMSE_valid: %.4f' % (loss, RMSE, RMSE_t))

    # draw the loss and RMSE
    plt.figure(figsize=(8, 6))
    plt.title('Loss')
    plt.plot(train_loss, label='train Loss', color='b')
    plt.plot(valid_loss, label='valid Loss', color='r')
    plt.legend()
    plt.savefig('Loss.png', dpi=200)

    plt.figure(figsize=(8, 6))
    plt.title('RMSE')
    plt.plot(train_RMSE, label='train RMSE', color='b')
    plt.plot(valid_RMSE, label='valid RMSE', color='r')
    plt.legend()
    plt.savefig('RMSE.png', dpi=200)

    np.save('final_params', W)


if __name__ == '__main__':
    run_baseline(split=True)
