import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

melbourne_file_path = 'melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
# print(melbourne_data.columns)

melbourne_data = melbourne_data.dropna(axis=0)

# Target
y = melbourne_data.Price
# print(y)

# choose features
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]
# print(X.describe())
# print(X.head())


# Building your model
# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1, ccp_alpha=0.0)

# Fit model
melbourne_model.fit(X, y)

# print("Making predictions for the following 5 houses:")
# print(X.head(), "\n")
# print("The predictions are")
# print(melbourne_model.predict(X.head()))
# print("\ny real values:")
# print(y.head())


predicted_home_prices = melbourne_model.predict(X)
print("\nAverage absolute error:", mean_absolute_error(y, predicted_home_prices))