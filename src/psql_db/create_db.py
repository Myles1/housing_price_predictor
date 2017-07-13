import pandas as pd
import numpy as np
from sqlalchemy import create_engine

PSQL_USER = 'MHollowed'
PSQL_PW = ''
PSQL_HOST = ''
PSQL_DB = 'real_estate_data'

def pandas_engine():
    '''Creates engine to access psql'''
    url = "postgresql://{}:{}@{}/{}".format(PSQL_USER, PSQL_PW, PSQL_HOST, PSQL_DB)
    return create_engine(url)

def create_table(df, table_name):
    '''Dumps pandas df to psql table'''
    df.to_sql(table_name, pandas_engine(), if_exists='append')

if __name__ == '__main__':
    '''Create a psql table from csv data file'''

    # table_name = 'property_info'
    # file_path = 'data/csvs/EXTR_ResBldg.csv'

    # table_name = 'sales_info'
    # file_path = 'data/csvs/EXTR_RPSale.csv'

    table_name = 'appraisal_history2'
    #file_path = 'data/csvs/EXTR_RealPropApplHist_V.csv'

    file_path = 'data/csvs/temp.csv'
    file_path2 = 'data/csvs/temp2.csv'

    dtypes = {
        'Major' : str,
        'Minor' : str,
        'RollYr' : np.int32,
        'RevalOrMaint' : str,
        'LandVal' : np.int32,
        'ImpsVal' : np.int32,
        'NewDollars' : np.int32,
        'SelectMethod' : np.int32,
        'SelectReason' : np.int32,
        'SelectAppr' : str,
        'SelectDate' : np.datetime64,
        'PostStatus' : np.int32,
        'PostDate' : np.datetime64,
        'UpdatedBy' : str,
        'UpdateDate' : np.datetime64
    }


    for path_ in [file_path, file_path2]:
        print("Reading {} into dataframe...".format(file_path))
        df = pd.read_csv(file_path, low_memory=False, names=dtypes.keys())

        print("Replacing key values...")
        df.replace(r'\s+', 0, regex=True, inplace=True)

        print("Creating {} table...".format(table_name))
        create_table(df, table_name)
