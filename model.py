import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from utils import plot_bar
from sklearn.metrics import confusion_matrix, recall_score, make_scorer
from datetime import datetime


def feature_engineering(df):
    '''
    Clean Pandas dataframe and reduce to necessary features
    '''
    df['last_trip_date'] = pd.to_datetime(df['last_trip_date'])
    df['signup_date'] =  pd.to_datetime(df['signup_date'])
    cut_off_date = datetime.strptime('2014-07-01', '%Y-%m-%d')
    churn_date = datetime.strptime('2014-06-01', '%Y-%m-%d')
    df['churn'] = (df['last_trip_date'] < churn_date).astype(int)
    # df['churn'] = df['churn'].apply(lambda x: x.day)
    df['days_until_cutoff'] = (cut_off_date - df['signup_date'])
    df['days_until_cutoff'].apply(get_days)
    df['days_until_cutoff'] = (df['days_until_cutoff'] / np.timedelta64(1, 'D')).astype(int)
    df.replace({False:0, True:1}, inplace=True)
    df = df.dropna(axis=0, how='any')
    df = pd.get_dummies(df)
    df.pop('last_trip_date')
    df.pop('signup_date')
    cols = [u'avg_rating_by_driver', u'avg_rating_of_driver',
       u'avg_surge', u'surge_pct', u'trips_in_first_30_days',
       u'luxury_car_user', u'weekday_pct',
       u'city_Astapor', u'city_King\'s Landing',
       u'phone_Android', u'phone_iPhone', 'churn']
    return df, cols

def gridsearch(model, scorer, param_grid, X, y):
    '''     
    Perform Gridsearch for the given model and return the best model
    '''
    grid_search = GridSearchCV(model, param_grid=param_grid, verbose=1, n_jobs=-1)
    grid_search.fit(X, y)
    print(grid_search.best_estimator_)
    print("Grid scores on development set:")
    return model

if __name__ == '__main__':
    train = pd.read_csv('../data/churn_train.csv')
    test = pd.read_csv('../data/churn_test.csv')

    #drop all the missing values
    train, cols = feature_engineering(train)

    #Seperating into results(y) and features(X)
    y = train.pop('churn').values
    X = train.values

    # perform training and validation split
    np.random.seed(0)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33)

    rf = RandomForestClassifier()
    recall = make_scorer(recall_score)
    param_grid = dict(n_estimators=[20, 40, 100, 200],
                  max_depth = [3,4,6,8],
                  max_features = ['sqrt', 'log2', None],
                  bootstrap = [True, False])
    rf = gridsearch(rf, recall, param_grid, X_train, y_train)
    rf.fit(X_train, y_train)
    print recall_score(rf.predict(X_val), y_val)
