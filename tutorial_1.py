## DOCUMENTATION ______________________________________________
''' Desc:
    Ref: https://www.analyticsvidhya.com/blog/2018/05/generate-accurate-forecasts-facebook-prophet-python-r/

'''


## Load Python Modules ---------------------------------------
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet
from sklearn.model_selection import train_test_split

## Project Modules -------------------------------------------
import functs_tutorial as m1


## Directories -----------------------------------------------
dir_data    = r'/home/cc2/Desktop/repositories/fb_prophet/data'

## Load Data -------------------------------------------------
afile           = r'traffic_train.csv'
data            = pd.read_csv(dir_data + '/' + afile)
train_sample    = int(0.7 * len(data['Count']))
train           = data.iloc[0: train_sample, :]
test            = data.iloc[train_sample : len(data['Count'])]

## Inspect Data -----------------------------------------------
#m1.inspect_data(train, test)
#m1.inspect_ts_info(train)


## Prepare Data -----------------------------------------------
train.loc[:, 'Datetime']   = pd.to_datetime(train.Datetime, format='%d-%m-%Y %H:%M')
test.loc[:,  'Datetime']   = pd.to_datetime(test.Datetime, format='%d-%m-%Y %H:%M')
train['hour']       = train.Datetime.dt.hour
train['day']        = train.Datetime.dt.day
train['month']      = train.Datetime.dt.month


## Visualizations ---------------------------------------------

def plot_passenger_cnt_date(train):
    fig = plt.figure()
    ax  = plt.axes()
    y   = train['Count']
    x   = train['Datetime']
    ax.plot(x, y)
    plt.pause(5)
    plt.close()


def plot_avg_passenger_cnt(train, col, pause=5):
    fig     = plt.figure()
    ax      = plt.axes()
    y_mu    = train.groupby(col)['Count'].mean()
    x       = y_mu.index
    ax.plot(x,y_mu)
    ax.set_title('Plot of Avg Count per {}'.format(col))
    plt.pause(pause)
    plt.close()


## Prophet -----------------------------------------------------
''' Data:   Prophet requires that the data be in the form 
            y   = Target
            ds  = Datetime
'''
def fit_prophet(train, test, pprint, plot, plot_forecast):
    # Prepare Data
    daily_train         = pd.DataFrame({})
    daily_test          = pd.DataFrame({})
    daily_train['ds']   = train['Datetime']
    daily_train['y']    = train['Count']
    daily_test['ds']    = test['Datetime']
    daily_test['y']     = test['Count']

    # Fit Model
    m           = Prophet(yearly_seasonality = True, seasonality_prior_scale=0.1)
    m.fit(daily_train)
    future      = m.make_future_dataframe(periods= len(test.iloc[:, 0]))
    forecast    = m.predict(future) 
    test_forecast   = forecast.loc[ len(daily_train.iloc[:, 0]) :, 'yhat']

    # Print Forecast Object
    if pprint==True:
        print(forecast)   

    if plot==True:
        m.plot_components(forecast)
        plt.pause(5)
        plt.close()
    if plot_forecast==True:

        plt.plot(daily_test['y'], color='red')
        plt.plot(test_forecast, color='blue')
        plt.show()

fit_prophet(train, test, False, False, True)
