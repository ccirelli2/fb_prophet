## DOCUMENTATION _______________________________________
''' Desc:   Functions for the pi4 tutorial
'''


# Modules
from fbprophet import Prophet


## Functions -------------------------------------------

def inspect_data(data):
    print(data.head())
    print(data.shape)
    print(data.dtypes)


def fit_n_forecast(data):
    m   = Prophet()
    m.fit(data.loc[:, ['ds', 'y']])

    # Make Prediction
    future      = m.make_future_dataframe(periods=365)
    forecast    = m.predict(future)

    return forecast

def inspect_forecast(forecast):
    # Print Forecast
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    # Plot Forecast
    fig1    = m.plot(forecast)
    plt.show()











