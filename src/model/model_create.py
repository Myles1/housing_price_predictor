# Database connect
import psycopg2

# Numpy arrays
import numpy as np

# DataFrames
import pandas as pd

# Models
from xgboost import XGBRegressor
# from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor#, GradientBoostingRegressor

# Cross validation
from sklearn.cross_validation import train_test_split#, cross_val_score
# from sklearn.ensemble.partial_dependence import plot_partial_dependence
# from sklearn.model_selection import KFold, ShuffleSplit
# from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, make_scorer
# make_scorer = make_scorer(median_absolute_error)

# Preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

# Persisting Model
import pickle


def execute_query(q):
    conn.rollback()
    cur.execute(q)
    conn.commit()
    return cur.fetchall(), cur.description

def filter_df(keep_values, col_name, df):
    mask = df[col_name].isin(keep_values)
    return df[mask]


class RealEstatePredictor(object):

    def __init__(self):
        self.model_XG = XGBRegressor(learning_rate=0.1, n_estimators=500, max_depth=15)
        self.model_RF = RandomForestRegressor(n_jobs=-1, n_estimators=250)
        self.ensemble = None

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y)

        self.model_XG.fit(X, y)
        self.model_RF.fit(X, y)

        self.ensemble = [self.model_XG, self.model_RF]

        return self

    def score(self, X_test, y_test):
        y_hat_XG = self.model_XG.predict(X_test)
        y_hat_RF = self.model_RF.predict(X_test)

        X_preds = [y_hat_XG, y_hat_RF]
        ensemble = XGBRegressor(learning_rate=0.1, n_estimators=500, max_depth=15)
        ensemble.fit(X_preds)

        return median_absolute_error(y_hat, y_test)

    def predict(self, X):




        return self.model.predict(X)





if __name__ == '__main__':
    '''Steps to create and persist Real Estate Predictor'''


    db_info = {
        'user':'app_connect',
        'database':'real_estate_data',
        'host' : 'localhost',
        'port' : 5432,
        'password' : 'flying_horse536'
    }
    print("Connecting to Database...")
    print("Database: {}".format(db_info['database']))
    print("User: {}".format(db_info['user']))
    print("Host: {}".format(db_info['host']))
    print("Port: {}".format(db_info['port']))
    conn = psycopg2.connect(**db_info)
    cur = conn.cursor()

    # Query database for all sales info, left joined with property info
    query = '''
        SELECT *
        FROM sales_info s
        LEFT JOIN property_info p
        ON s.major = p.major AND s.minor = p.minor
        ;
    '''

    print("Sending query")
    query_result, description = execute_query(query)
    column_names = [d.name for d in description]

    print("Creating DataFrame")
    df = pd.DataFrame(query_result, columns=column_names)




    print("Filtering values")
    '''
    PrincipalUse value of 6 represents Residential buildings
    Keep only sales with PrincipalUse value of 6
    '''
    principal_use_keep_values = [6]
    df = filter_df(keep_values=principal_use_keep_values, col_name='principal_use', df=df)

    '''
    PropertyType value of 1 represents Land Only
    PropertyType value of 2 represents Land with New Building
    PropertyType value of 3 represents Land with Previously Used Building
    Keep only sales with PropertyType value in [1, 2, 3]
    '''
    property_type_keep_values = [1, 2, 3]
    df = filter_df(property_type_keep_values, 'property_type', df)

    '''
    SaleInstrument value of 3 represents a Statutory Warranty Deed
    By using this deed, the seller promises the buyer
    1. The seller is the owner of the property and has the right to sell it
    2. No one else is possessing the property
    3. There are no encumbrances against the property
    4. No one with a better claim to the property will interfere with the transferee’s rights
    5. The seller will defend certain claims regarding title to the property
    '''
    sale_instrument_keep_values = [3]
    df = filter_df(sale_instrument_keep_values, 'sale_instrument', df)


    print("Converting 'document_date' to DateTime Format")
    # Change 'DocumentDate' to DateTime
    df['document_date'] = pd.to_datetime(df['document_date'])

    print("Creating 'sale_year' and 'sale_month' columns")
    # Get SaleYear and SaleMonth columns out of the DateTime object
    df['sale_year'] = pd.DatetimeIndex(df['document_date']).year
    df['sale_month'] = pd.DatetimeIndex(df['document_date']).month

    print("Removing years prior to 1995")
    '''
    Remove years before 1995
    '''
    df = df[df['sale_year'] > 1994]

    ### Uncomment to select specific years ###
    #df = df[df['sale_year'] > 2011]


    print("Shuffling DataFrame")
    # Shuffle dataframe
    df = shuffle(df)

    print("Filling NaNs")
    df.fillna(0, inplace=True)




    # Setting up features for model:

    # Features:
    sales_feature = [
        'property_type',
        'sale_reason',
        'property_class',
        'sale_year',
        'sale_month'
    ]

    property_features = [
        'nbr_living_units',
        'stories',
        'bldg_grade',
        'bldg_grade_var',
        'sq_ft_1st_floor',
        'sq_ft_half_floor',
        'sq_ft_2nd_floor',
        'sq_ft_upper_floor',
        'sq_ft_unfin_full',
        'sq_ft_unfin_half',
        'sq_ft_tot_living',
        'sq_ft_tot_basement',
        'sq_ft_fin_basement',
        'fin_basement_grade',
        'sq_ft_garage_basement',
        'sq_ft_garage_attached'
    ]

    combined_features = sales_feature + property_features

    features_as_is = [
        'sale_year',
        'sale_month'
    ]

    features_to_scale = [
        'sq_ft_1st_floor',
        'sq_ft_half_floor',
        'sq_ft_2nd_floor',
        'sq_ft_upper_floor',
        'sq_ft_unfin_full',
        'sq_ft_unfin_half',
        'sq_ft_tot_living',
        'sq_ft_tot_basement',
        'sq_ft_fin_basement',
        'sq_ft_garage_basement',
        'sq_ft_garage_attached'
    ]

    dummy_features = [
        'property_type',
        'sale_reason',
        'property_class',
        'nbr_living_units',
        'stories',
        'bldg_grade',
        'bldg_grade_var',
    ]


    print("Scaling columns:")
    for col in features_to_scale:
        print(col)
    # Standardize the dataframe
    scalar = StandardScaler().fit(df[features_to_scale])
    df[features_to_scale] = scalar.transform(df[features_to_scale])

    final_df = df[features_as_is + features_to_scale + ['sale_price']]

    print("Creating dummy columns:")
    for col in dummy_features:
        print(col)
    # Get dummy cols
    dummies = pd.get_dummies(df[dummy_features].applymap(str))
    final_df = pd.concat([final_df, dummies], axis=1)

    print("Finalizing DataFrame")
    final_df = shuffle(final_df)

    print("Collecting target values")
    # Target value: SalePrice
    y = final_df.pop('sale_price')
    X = final_df


    print("Initializing model")
    model = RealEstatePredictor()

    print("Fitting model")
    model.fit(X, y)


    print("Pickling model")


    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("Finished")
