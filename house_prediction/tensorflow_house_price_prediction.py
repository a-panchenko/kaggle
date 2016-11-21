import numpy as np
import pandas as pd
import tensorflow as tf


def clear_data(df):
    df['BsmtFinSF1'].fillna(df['BsmtFinSF1'].median(), inplace=True)
    df['BsmtFinSF2'].fillna(df['BsmtFinSF2'].median(), inplace=True)
    df['BsmtUnfSF'].fillna(df['BsmtUnfSF'].median(), inplace=True)
    df['BsmtFinType1'].fillna(df['BsmtFinType1'].mode().values[0], inplace=True)
    df['BsmtFinType1'].fillna(df['BsmtFinType2'].mode().values[0], inplace=True)
    df['GarageYrBlt'].fillna(df['GarageYrBlt'].mode().values[0], inplace=True)
    df['LotFrontage'].fillna(df['LotFrontage'].median(), inplace=True)
    df['MasVnrType'].fillna('None', inplace=True)
    df['MasVnrArea'].fillna(0, inplace=True)
    df['Electrical'].fillna(df['Electrical'].mode().values[0], inplace=True)
    df['MSSubClass'] = df['MSSubClass'].astype(str)
    df['OverallCond'] = df['OverallCond'].astype(str)
    df['MoSold'] = df['MoSold'].astype(str)
    df['TotalBsmtSF'].fillna(0, inplace=True)
    df['BsmtFullBath'].fillna(0, inplace=True)
    df['BsmtHalfBath'].fillna(0, inplace=True)
    df['GarageCars'].fillna(0, inplace=True)
    df['GarageArea'].fillna(0, inplace=True)
    return pd.get_dummies(df)


if __name__ == '__main__':
    EPOCHS = 100000
    PRINT_STEP = 1000

    raw_train_df = pd.read_csv('data/train.csv')
    raw_test_df = pd.read_csv('data/test.csv')
    prices = np.reshape(raw_train_df['SalePrice'].values, (1460, 1))
    df = pd.concat([raw_train_df.drop('SalePrice', axis=1), raw_test_df])
    df = clear_data(df)
    train = df.values[0:1460, :]
    test = df.values[1460:, :]
    # Build a RNN
    x_ = tf.placeholder(dtype=tf.float32, shape=[None, train.shape[1] - 1])
    y_ = tf.placeholder(dtype=tf.float32, shape=[None, 1])

    cell = tf.nn.rnn_cell.BasicRNNCell(num_units=train.shape[1] - 1)

    outputs, states = tf.nn.rnn(cell, [x_], dtype=tf.float32)
    outputs = outputs[-1]

    W = tf.Variable(tf.random_normal([train.shape[1] - 1, 1]))
    b = tf.Variable(tf.random_normal([1]))

    y = tf.matmul(outputs, W) + b

    cost = tf.reduce_mean(tf.square(y - y_))
    rmse = tf.sqrt(cost)
    train_op = tf.train.RMSPropOptimizer(0.005, 0.2).minimize(cost)

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for i in range(EPOCHS):
            sess.run(train_op, feed_dict={x_: train[:, 1:], y_: prices})
            if i % PRINT_STEP == 0:
                c = sess.run(cost, feed_dict={x_: train[:, 1:], y_: prices})
                r = sess.run(rmse, feed_dict={x_: train[:, 1:], y_: prices})
                print('training cost:', c)
                print('rmse: ', r)

        response = sess.run(y, feed_dict={x_: test[:, 1:]})
        response = np.reshape(response, (1459,))
        predicted_matrix = np.c_[[raw_test_df.values[:,0].astype(int), response.astype(int)]]
        predicted_df = pd.DataFrame(predicted_matrix.T, columns=['Id', 'SalePrice'])
        predicted_df.to_csv('results/house_price_1-1.csv', index=False)