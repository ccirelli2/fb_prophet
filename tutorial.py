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
afile   = r'traffic_train.csv'
bfile   = r'traffic_test.csv'
train   = pd.read_csv(dir_data + '/' + afile)
test    = pd.read_csv(dir_data + '/' + bfile)


## Inspect Data -----------------------------------------------
#m1.inspect_data(train, test)
#m1.inspect_ts_info(train)


## Prepare Data -----------------------------------------------
train['Datetime']   = pd.to_datetime(train.Datetime, format='%d-%m-%Y %H:%M')
test['Datetime']    = pd.to_datetime(test.Datetime, format='%d-%m-%Y %H:%M')
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
def fit_prophet(train, test):
    # Prepare Data
    daily_train         = pd.DataFrame({})
    daily_train['ds']   = train['Datetime']
    daily_train['y']    = train['Count']

    # Fit Model
    m           = Prophet(yearly_seasonality = True, seasonality_prior_scale=0.1)
    m.fit(daily_train)
    future      = m.make_future_dataframe(periods=213)
    forecast    = m.predict(future) 
    print(forecast)
    # looks like yhat is the prediction column name
    # Visualize Components
    #m.plot_components(forecast)
    #plt.show()


