# import pandas as pd
#
# from sklearn.preprocessing import OneHotEncoder
# from scipy.sparse import hstack
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error, mean_squared_error
#
# # load the data
# df = pd.read_csv("live_supermarket_data.csv")
#
# # encode the Category and Product Name columns
# encoder = OneHotEncoder()
# X_category = encoder.fit_transform(df[['Category']])
# X_product_name = encoder.fit_transform(df[['Product Name']])
#
# # concatenate the encoded arrays
# X = hstack([X_category, X_product_name])
#
# # select the target
# y = df['Qty']
#
# # split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # fit a linear regression model
# model = LinearRegression()
# model.fit(X_train, y_train)
#
# # make predictions on the testing data
# y_pred = model.predict(X_test)
#
# # get the index of the highest selling product
# max_index = y_pred.argmax()
#
# # get the corresponding product name
# product_name = df['Product Name'].iloc[max_index]
#
# # print the predicted product and quantity
# print(f"Predicted high selling product: {product_name}")
# print(f"Predicted quantity: {y_pred[max_index]:.2f}")
#
# # evaluate the model using mean absolute error and mean squared error
# mae = mean_absolute_error(y_test, y_pred)
# mse = mean_squared_error(y_test, y_pred)
#
# print(f"Mean Absolute Error: {mae:.2f}")
# print(f"Mean Squared Error: {mse:.2f}")

# import necessary libraries
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# load the data
df = pd.read_csv("live_supermarket_data.csv")

# encode the Category and Product Name columns
encoder = OneHotEncoder()
X_category = encoder.fit_transform(df[['Category']])
X_product_name = encoder.fit_transform(df[['Product Name']])

# concatenate the encoded arrays
X = hstack([X_category, X_product_name])

# select the target
y = df['Qty']

# get the date for which you want to predict the high selling product
date = '2022-01-26'

# filter the data to include only rows for the given date
df_filtered = df[df['Time of Purchase'].str.startswith(date)]

# extract the features and target from the filtered data
X_filtered = X[df_filtered.index]
y_filtered = y[df_filtered.index]

# split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42)

# fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# make predictions on the testing data
y_pred = model.predict(X_test)

# get the index of the highest selling product
max_index = y_pred.argmax()

# get the corresponding product name
product_name = df_filtered['Product Name'].iloc[max_index]

new_data = {
    'Category': 'Unknown',
    'Time of Purchase': 'Unknown'
}

X_new = encoder.transform(pd.DataFrame(new_data, index=[0]))

prediction = model.predict(X_new)


# print the predicted product and quantity
print(f"Predicting for date: {date}")
print(f"Predicted high selling product: {product_name}")
print(f"Predicted quantity: {y_pred[max_index]:.2f}")
