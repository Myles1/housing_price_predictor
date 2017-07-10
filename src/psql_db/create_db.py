import pandas as pd
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
    df.to_sql(table_name, pandas_engine())

if __name__ == '__main__':
    '''Create a psql table from csv data file'''

    # table_name = 'property_info'
    # file_path = 'data/csvs/EXTR_ResBldg.csv'

    table_name = 'sales_info'
    file_path = 'data/csvs/EXTR_RPSale.csv'

    print("Reading {} into dataframe...".format(file_path))
    df = pd.read_csv(file_path, low_memory=False)
    df.replace(r'\s+', 0, regex=True, inplace=True)




    print("Creating {} table...".format(table_name))
    create_table(df, table_name)
