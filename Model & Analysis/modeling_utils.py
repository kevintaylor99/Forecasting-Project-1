import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from scipy import signal
from math import *

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

def train_test_split(df, n_test):
    """
    Split the train and test data, maintaining the order
    """
    return df[:-n_test], df[-n_test:]

def MAPE(actual, predicted):
    """
    Measure the mean absolute percentage error (MAPE)
    """
    return sqrt(abs(actual-predicted)/actual)

def walk_forward_validation_sarimax(df, model, n_test):
    """
    Perform walk-forward validation with a defined n_test in the data
    """
    train, test = train_test_split(df, n_test)
    model = model.fit(max_iter = 20, method = 'powell')
    # walk forward
    predictions = model.forecast(n_test)
    predictions.append(yhat) #store the forecast
    # estimate error
    return predictions, MAPE(test.value, predictions)

