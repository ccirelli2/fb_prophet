## DOCUMENTATION ______________________________________________
''' Desc:
    Ref: https://www.analyticsvidhya.com/blog/2018/05/generate-accurate-forecasts-facebook-prophet-python-r/

'''


## Load Python Modules ---------------------------------------
import pandas as pd
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



## Prepare Data -----------------------------------------------
train['Datetime']   = pd.to_datetime(train.Datetime, format='%d-%m-%Y %H:%M')
test['Datetime']    = pd.to_datetime(test.Datetime, format='%d-%m-%Y %H:%M')
train['hour']       = train.Datetime.dt.hour

def plot_passenger_count(train):
    fig = plt.figure()
    ax  = plt.axes()
    y   = train['Count']
    x   = train['Datetime']
    ax.plot(x, y)
    plt.pause(5)
    plt.close()

