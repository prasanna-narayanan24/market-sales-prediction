import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import numpy as np

def train_model():
    # Load and preprocess the data
    data = pd.read_csv("live_supermarket_data.csv")
    data['Time of Purchase'] = pd.to_datetime(data['Time of Purchase'])
    data = data.groupby(['Time of Purchase', 'Product Name']).agg({'Qty': 'sum'}).reset_index()

    # Convert 'Product Name' column to numerical values
    le = LabelEncoder()
    data['Product Name'] = le.fit_transform(data['Product Name'])

    # convert 'Time of Purchase' to numerical representation
    data['Time of Purchase'] = data['Time of Purchase'].astype(int) / 10 ** 9

    # Split the data into training and test sets
    train_data, test_data = train_test_split(data, test_size=0.2)

    # Create and fit the model
    model = LinearRegression()
    X_train = train_data[['Time of Purchase', 'Product Name']]
    y_train = train_data['Qty']

    X_test = test_data[['Time of Purchase', 'Product Name']]
    Y_test = test_data['Qty']

    model.fit(X_train, y_train)

    # Predict the maximum quantity for a given future date and product name
    future_date = pd.to_datetime("2021-08-30")
    product_name = "Bagels"
    product_name_num = le.transform([product_name])
    future_date = future_date.timestamp()

    X_new = pd.DataFrame({
        'Time of Purchase': future_date,
        'Product Name': product_name_num
    })
    prediction = model.predict(X_new)
    print("Maximum predicted quantity for {}: {}".format(product_name, prediction[0]))

    score = model.score(X_test, Y_test)
    print(f'score: {score}')

    print(prediction)

def test_model():
    # Load and preprocess the data
    data = pd.read_csv("live_supermarket_data.csv")
    data['Time of Purchase'] = pd.to_datetime(data['Time of Purchase'])
    data = data.groupby(['Time of Purchase', 'Product Name']).agg({'Qty': 'sum'}).reset_index()

    # Convert 'Product Name' column to numerical values
    le = LabelEncoder()
    data['Product Name'] = le.fit_transform(data['Product Name'])

    # convert 'Time of Purchase' to numerical representation
    data['Time of Purchase'] = data['Time of Purchase'].astype(int) / 10 ** 9

    # Split the data into training and test sets
    train_data, test_data = train_test_split(data, test_size=0.1)

    # Create and fit the model
    model = LinearRegression()
    X_train = train_data[['Time of Purchase', 'Product Name']]
    y_train = train_data['Qty']
    model.fit(X_train, y_train)

    # Predict the maximum quantity for a given future date and product name
    future_date = pd.to_datetime("2021-08-30")
    product_name = "Bagels"
    product_name_num = le.transform([product_name])
    future_date = future_date.timestamp()

    X_new = pd.DataFrame({
        'Time of Purchase': future_date,
        'Product Name': product_name_num
    })
    prediction = model.predict(X_new)
    print("Maximum predicted quantity for {}: {}".format(product_name, prediction[0]))

    print(prediction)


train_model()
# print("Give Date: Jan 15, 2023")
# print("Highest Selling Product Thus far: Grapes")
# print("Estimated Selling Qty: 250")


# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.preprocessing import LabelEncoder
#
# # Load the data
# data = pd.read_csv("live_supermarket_data.csv")
# data['Time of Purchase'] = pd.to_datetime(data['Time of Purchase'])
#
# # Group the data by 'Time of Purchase' and 'Product Name' and calculate the total quantity
# data = data.groupby(['Time of Purchase', 'Product Name']).agg({'Qty': 'sum'}).reset_index()
#
# # Encode product names to numerical values
# le = LabelEncoder()
# data['Product Name'] = le.fit_transform(data['Product Name'])
#
# # Create a scatter plot
# plt.scatter(data['Time of Purchase'], data['Qty'], c=data['Product Name'], cmap='rainbow')
# plt.xlabel('Time of Purchase')
# plt.ylabel('Total Quantity')
# plt.title('Total Quantity of Products over time')
# plt.show()



