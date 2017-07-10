import pandas as pd
from sklearn.ensemble import RandomForestRegressor




def get_create_df(file_path, nrows=10000):
    df = pd.read_csv(file_path, nrows=nrows)
    df = convert_col_to_datetime('DocumentDate', df)
    df = create_month_year_cols('DocumentDate', df)
    return df

def convert_col_to_datetime(col_name, df):
    df[col_name] = pd.to_datetime(df[col_name])

def create_month_year_cols(col_name, df):
    df['SaleMonth'] = pd.DatetimeIndex(df[col_name]).month
    df['SaleYear'] = pd.DatetimeIndex(df[col_name]).year
    return df

def filter_df(keep_values, col_name, df):
    mask = df[col_name].isin(keep_values)
    return df[mask]



if __name__ == '__main__':
    '''Steps to create and persist Real Estate Predictor'''
    file_path = 'data/csvs/EXTR_RPSale.csv'
    df = pd.read_csv(file_path, nrows=10000)

    print(df.info())
