## DOCUMENTATION _______________________________________
''' Desc:   Functions for the pi4 tutorial
'''




## Python Modules --------------------------------------




## Functions -------------------------------------------


def inspect_data(train, test):
    ''' Desc:   function to return key metrics
                of the datasets. 
    '''
    dim_train   = train.shape
    dim_test    = test.shape
    cols        = train.columns
    dtype       = train.dtypes
    desc_train  = train.describe()
    desc_test   = test.describe()
    head        = train.head()
    print('Infromation about Trian & Test Data\n')
    print('Dimension train  => {}'.format(dim_train))
    print('Dimension test   => {}'.format(dim_test))
    print('Columns          => {}\n'.format(cols))
    print('Datatypes        =>\n {}\n'.format(dtype))
    print('Description train =>\n {}\n'.format(desc_train))
    print('Description test  =>\n {}\n'.format(desc_test))
    print('Dataframe head    =>\n {}'.format(head))

    return None


def inspect_ts_info(train):
    train['Datetime']   = pd.to_datetime(train.Datetime, format='%d-%m-%Y %H:%M')
    train['years']      = train.Datetime.dt.year
    train['hour']       = train.Datetime.dt.hour
    train['day']        = train.Datetime.dt.day
    train['month']      = train.Datetime.dt.month
    months              = train.groupby('month')['Count'].mean()
    days                = train.groupby('day')['Count'].mean()
    hours               = train.groupby('hour')['Count'].mean()
    yrs = train.groupby('years')['Count'].mean()
    print('\n**Get Time Series Information **\n')
    print('Range of Years => {}'.format(yrs.index.values))
    print('Range of Months => {}'.format(months.index.values))
    print('Range of days  => {}'.format(days.index.values))
    print('Range of hours => {}'.format(hours.index.values))

