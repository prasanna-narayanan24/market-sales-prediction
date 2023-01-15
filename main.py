import numpy as np
import random
import time

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

# one-hot encode the Category column
encoder = OneHotEncoder()

categories = ['Fruits', 'Vegetables', 'Bakery', 'Dairy', 'Meat', 'Seafood']


def predict_product(model: LinearRegression):
    # create a new data point with the given date
    date = '2022-01-26'
    new_data_point = encoder.transform(pd.DataFrame({'Category': ['Fruits'], 'Product Name': ['Bananas']})).toarray()
    new_data_point = np.concatenate((new_data_point, [[datetime.strptime(date, '%Y-%m-%d').toordinal()]]), axis=1)

    # make a prediction using the trained model
    prediction = model.predict(new_data_point)

    # print the prediction
    print(prediction)


def train_model():
    # load the data into a DataFrame
    df = pd.read_csv('live_supermarket_data.csv')

    X = encoder.fit_transform(df[['Category']]).toarray()

    # add the Time of Purchase column to the features
    X = np.concatenate((X, df['Time of Purchase'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').toordinal()).values.reshape(-1, 1)), axis=1)

    # select the target
    y = encoder.fit_transform(df[['Product Name']]).toarray()

    # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # create the model
    model = LinearRegression()

    # fit the model to the training data
    model.fit(X_train, y_train)

    # fit the OneHotEncoder on the entire dataset
    encoder.fit(df[['Category', 'Product Name']])

    # make predictions on the test set
    predictions = model.predict(X_test)

    # calculate the mean absolute error and mean squared error
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)

    # print the evaluation metrics
    print(f'MAE: {mae:.2f}')
    print(f'MSE: {mse:.2f}')

    return model


# _model = train_model()
# predict_product(_model)

# load the data into a DataFrame
df = pd.read_csv('live_supermarket_data.csv')

# one-hot encode the Category and Product Name columns
# encoder = OneHotEncoder()
transformer = ColumnTransformer(transformers=[('one_hot', OneHotEncoder(), ['Category', 'Product Name'])], remainder='passthrough')
X = transformer.fit_transform(df)

# convert the Time of Purchase column to a numeric value
X = np.concatenate((X, df['Time of Purchase'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S').toordinal()).values.reshape(-1, 1)), axis=1)

# select the target
# select the target
y = encoder.fit_transform(df[['Product Name']]).toarray()[:, 0]

# split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# create the model
model = LogisticRegression(multi_class='ovr')

# fit the model to the training data
model.fit(X_train, y_train)

# create a new data point with the given date
date = '2022-01-26'

feature_names_in_ = transformer.named_transformers_['one_hot'].get_feature_names(['Category', 'Product Name'])
# create a new data point with the same input feature names and order as the training data
new_data_point = pd.DataFrame(columns=feature_names_in_)
new_data_point = new_data_point.append({'Category_Fruits': 1, 'Product Name_Bananas': 1, 'Time of Purchase': datetime.strptime(date, '%Y-%m-%d').toordinal()}, ignore_index=True)

# # get the list of feature names
# feature_names = encoder.get_feature_names_out(['Category', 'Product Name', 'Time of Purchase'])
#
# # create a new data point with the same feature names as the training data
# new_data_point = pd.DataFrame(columns=feature_names)
# new_data_point = np.concatenate((new_data_point, [[datetime.strptime(date, '%Y-%m-%d').toordinal()]]), axis=1)

# make a prediction using the trained model
prediction = model.predict(new_data_point)

# print the prediction
print(prediction)


def create_data_set():
    # create list of product categories

    # create list of products
    products = []
    fruits = ['Apples', 'Bananas', 'Oranges', 'Strawberries', 'Grapes', 'Watermelon']
    vegetables = ['Carrots', 'Peas', 'Corn', 'Spinach', 'Lettuce', 'Tomatoes']
    bakery = ['Bread', 'Bagels', 'Pastries', 'Croissants', 'Donuts', 'Cookies']
    dairy = ['Milk', 'Yogurt', 'Cheese', 'Butter', 'Ice Cream', 'Cottage Cheese']
    meat = ['Beef', 'Pork', 'Chicken', 'Turkey', 'Lamb', 'Sausage']
    seafood = ['Salmon', 'Tuna', 'Shrimp', 'Crab', 'Lobster', 'Clams']
    products.extend(fruits)
    products.extend(vegetables)
    products.extend(bakery)
    products.extend(dairy)
    products.extend(meat)
    products.extend(seafood)

    products = random.sample(products, len(products))

    # create list of dates
    start_date = '2020-01-01'
    end_date = '2022-12-01'
    dates = pd.date_range(start_date, end_date)

    # create dataframe with columns
    df = pd.DataFrame(columns=['Category', 'Product Name', 'Time of Purchase', 'Qty', 'Customer Id'])

    def get_category_for_product(p: str) -> str:
        if p in fruits:
            return 'Fruits'
        if p in vegetables:
            return 'Vegetables'
        if p in bakery:
            return 'Bakery'
        if p in dairy:
            return 'Dairy'
        if p in meat:
            return 'Meat'
        if p in seafood:
            return 'Seafood'

    # fill dataframe with random data
    customer_id = 1
    rows = []
    for date in dates:
        for product in products:
            category = get_category_for_product(product)

            qty = random.randint(1, 10)
            timestamp = int(time.time())
            time_of_purchase = pd.to_datetime(date).strftime('%Y-%m-%d') + ' ' + str(random.randint(0, 23)).zfill(
                2) + ':' + str(random.randint(0, 59)).zfill(2) + ':' + str(random.randint(0, 59)).zfill(2)
            customer_gender = random.choice(['Male', 'Female'])
            rows.append({
                'Category': category,
                'Product Name': product,
                'Time of Purchase': time_of_purchase,
                'Qty': qty,
                'Customer Id': random.randint(1, 100000),
            })
            customer_id += 1

    df = pd.concat([df, pd.DataFrame(rows)])

    # save data to csv file
    df.to_csv('live_supermarket_data.csv', index=False)


