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

