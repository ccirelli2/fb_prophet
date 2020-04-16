## DOCUMENTATION ____________________________________________________
''' Desc:   Facebook's official tutorial to Prophet
    Ref:    https://facebook.github.io/prophet/docs/quick_start.html

    Input:  Alway two columns, ds and y.
            -   The ds is the datestampt and should be
                of the format expected by Pandas YYYY-MM-DD for a date and
                YYYY-MM-DD HH:MM:SS for a timestamp.
            -   The y column must be numerica and represents the measurement
                we wish to forecast.
    Add Regressors:
    https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html

    TowardDataScience - Prophet tutorial
    https://towardsdatascience.com/a-quick-start-of-time-series-forecasting-with-a-practical-example-using-fb-prophet-31c4447a2274
'''

## Python Packages -------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation


## Directories -----------------------------------------------------
dir_data = r'~/Desktop/repositories/fb_prophet/data'


## Data ------------------------------------------------------------
''' Desc:   Data represents the number of daily page views for the
            wikipedia page for Peyton Manning
'''
afile = r'manning.csv'
data = pd.read_csv(dir_data + '/' + afile)


## Inspect Data ----------------------------------------------------
def inspect_data(ndata):
    print(ndata.head())
    print(ndata.shape)
    print(ndata.dtypes)


## Preprocess Data ------------------------------------------------
data.loc[:, 'ds'] = pd.to_datetime(data.loc[:, 'ds'], format='%Y-%m-%d')
data['year'] = data.ds.dt.year
data['month'] = data.ds.dt.month
data['day'] = data.ds.dt.day




## Fit and Forecast Model --------------------------------------
''' Desc:   Predictions are made on a dataframe with a column ds containing
            the dates for which a prediction is to be made.
    Help:   Prophet has a help function called make_future_dataframe(periods=)
            that will help with creating this dataframe.
    Input:  col1 = DS, col2 = y.  This is the required format.
'''

def fit_forecast(data_):
    # Define data set
    df = data_.loc[:, ['ds', 'y']]
    # Instantiate Model
    m = Prophet()
    # Fit Model to data
    m.fit(df)


## Generate Prediction --------------------------------------------
''' Predictions are made on a dataframe with a column ds containing
    dates for which a prediction is to be made.

    Prophet provides a function to generate this data frame and is called
    Prophet.make_future_dataframe(periods='').
    The period indicates for how many days you want to generate the forecast.

    Note that you can also create an in sample forecast by passing historical dates.
'''

# Make Future Data Frame
'''
future = m.make_future_dataframe(periods=365)
'''

# Inspect Tail of Original DataFrame and Future DataFrame
'''
print(df.tail())
print(future.tail())
'''

# Generate a prediction
''''Forecast output columns:
        'ds', 'trend', 'yhat_lower', 'yhat_upper', 'trend_lower', 'trend_upper',
        'additive_terms', 'additive_terms_lower', 'additive_terms_upper',
        'weekly', 'weekly_lower', 'weekly_upper', 'yearly', 'yearly_lower',
        'yearly_upper', 'multiplicative_terms', 'multiplicative_terms_lower',
        'multiplicative_terms_upper', 'yhat'
'''

'''
forecast = m.predict(future)
trend = forecast.loc[:, 'trend']
'''

# Plot Forecast
'''
fig1 = m.plot(forecast)
fig2_components = m.plot_components(forecast)
'''





# Generate In Sample Prediction -----------------------------------

# Train Test Split
sample = round(0.7 * len(data.iloc[:, 1]))
df_train = data.loc[0: sample, ['ds','y']]
df_test = data.loc[sample : , ['ds','y']]

# Instantiate Model & Fit

m = Prophet()
m.fit(df_train)

# Make a future dataframe
future = m.make_future_dataframe(periods= len(df_test.iloc[:, 0]))

# Make a prediction
pred = m.predict(future)

df_pred_vs_actual = pd.DataFrame({})
df_pred_vs_actual['pred'] = pred.loc[sample:, 'yhat']
df_pred_vs_actual['actual'] = df_test.loc[:, 'y']


df_pred_vs_actual.plot()
plt.show()
plt.title('Actual vs Prediction')


