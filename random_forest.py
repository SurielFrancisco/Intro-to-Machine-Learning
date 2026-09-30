import pandas as pd

# Load data
melbourne_file_path = 'melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# Filter rows with missing values
melbourne_data = melbourne_data.dropna(axis=0)

# Choose target and features
y = melbourne_data .Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                        'YearBuilt', 'Lattitude', 'Longtitude']
X = melbourne_data[melbourne_features]

from sklearn.model_selection import train_test_split

# Split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to 
# the random_state argument guarantees we get the same split every time we
# run this code.
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(X_train, y_train)
melb_preds = forest_model.predict(X_valid)
print(mean_absolute_error(y_valid, melb_preds))