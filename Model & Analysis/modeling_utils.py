import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from scipy import signal
from math import *
from statsmodels.tsa.statespace.sarimax import SARIMAX

def create_kwh_dataframe(df):
    """
    Processes the raw weekly usage data for an account
    into useable ts data to fit a model

    Parameters
    ----------
    df (pd.DataFrame) - pandas DataFrame of weekly usage

    Returns
    -------
    df_out (pd.DataFrame) - pandas DataFrame with Datetime as index
    """
    df_out = df.copy(deep = True)
    df_out.drop(df_out[df_out.value==0].index, inplace = True) # for accounts not yet created, their values are 0
    df_out.set_index( # set the index as a datetime from year and week
        pd.to_datetime([datetime.date.fromisocalendar(year, week, 1) for year, week in zip(df_out.year, df_out.week)]),
        inplace = True
    )
    df_out.drop(columns = ['t', 'week', 'year'], inplace = True) # drop the rest of the columns other than the value in kwh
    df_out.drop(min(df_out.index), inplace = True) # drop the first week, may be incomplete data
    df_out.drop(max(df_out.index), inplace = True) # drop the last week, may be incomplete data
    return df_out

def ts_plots(df, auto_lags):
    """
    Plots the timeseries, autocorrelation and PSD and prints the highest PSD period

    Parameters
    ----------
    df (pd.DataFrame) - pandas DataFrame with a DatetimeIndex and a column named 'value'
    auto_lags (int) - the number of autocorrelation lags to plot, maxlags parameter in ax.acorr function

    Returns
    -------
    """
    # Overall ts plot
    plt.figure(figsize=(10, 7))
    fig, ax = plt.subplots()
    plt.plot(df.value)
    plt.ylabel('Value')
    plt.xlabel('Time t')
    plt.legend(['Value'])
    plt.grid(True)
    plt.show()

    # Autocorrelation
    plt.figure(figsize=(10, 7))
    fig, ax = plt.subplots()
    ax.acorr(df.value.values ,maxlags = auto_lags)
    plt.grid(which='minor')
    plt.legend(['$R_X$(\u03C4)'],loc='upper left')
    plt.xlabel('\u03C4')
    ax.grid(True, which='both')
    plt.tight_layout()
    plt.show()

    # Overall PSD for all accounts combined
    freqs, psd = signal.welch(df.value.values)
    fig, ax = plt.subplots()
    ax.plot(freqs, psd)
    plt.grid(which='minor')
    plt.legend(['PSD'])
    plt.xlabel('Frequency')
    plt.ylabel('Power')
    ax.grid(True, which='both')
    plt.tight_layout()
    plt.show()
    print('Max Power at period {}'.format(1/freqs[psd == max(psd)][0]))
    return

def train_test_split(df, n_test, n_validation):
    """
    Split the train and test data, maintaining the order
    Splits the test set into three equal parts, returns all 5 DataFrames

    Parameters
    ----------
    df (pd.DataFrame) - input timeseries DataFrame
    n_test (int) - number of test points
    n_validation (int) - number of validation points

    Returns
    -------
    Five DataFrames corresponding to train, validation, and test 1, 2, 3
    """
    test_set = df[-n_test:]
    test_thirds = round(1/3*len(test_set))
    return df[:-n_validation-n_test], df[-n_validation-n_test:-n_test], test_set[:-2*test_thirds], test_set[-2*test_thirds:-test_thirds], test_set[-test_thirds:]

def MAPE(actual, predicted):
    """
    Measure the mean absolute percentage error (MAPE)
    """
    return np.mean(np.abs(actual-predicted)/actual)

def walk_forward_validation_sarimax(df, n_test, n_validation, p, d, q, s_p, s_d, s_q, seasonality, type = 'valid'):
    """
    Perform walk-forward validation with a defined n_test in the data

    Parameters
    ----------
    df (pd.DataFrame) - input timeseries DataFrame
    n_test (int) - number of test points
    n_validation (int) - number of validation points
    p, d, q, s_p, s_d, s_q, seasonality (int) - SARIMAX parameters
    type - specifies validation or test predictions, which to output
    """
    train, valid, test1, test2, test3 = train_test_split(df, n_test, n_validation)
    l1, l2, l3 = len(test1), len(test2), len(test3)
    if type == 'valid':
        model = SARIMAX( # validation case - train on training set only
            train,
            order = (p, d, q),
            seasonal_order = (s_p, s_d, s_q, seasonality)
        )
        model_fit = model.fit(max_iter = 20, method = 'powell', disp = False)
        predictions = model_fit.forecast(len(valid))
        # estimate error
        return train, valid, model_fit.predict(), predictions
    else:
        model = SARIMAX( # test case - train on training and validation set 
            train.append(valid),
            order = (p, d, q),
            seasonal_order = (s_p, s_d, s_q, seasonality)
        )
        model_fit = model.fit(max_iter = 20, method = 'powell', disp = False)
        # walk forward
        predictions = model_fit.forecast(len(test1) + len(test2) + len(test3))
        # estimate error
        return train.append(valid), [test1, test2, test3], model_fit.predict(), [predictions[:l1], predictions[l1:l1+l2], predictions[l1+l2:]]