import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

data = pd.read_csv(r'C:\Users\GowthamDurai\Desktop\Office\AQI\air-quality-india.csv')

data['Timestamp'] = pd.to_datetime(data['Timestamp'])

data['Hour'] = data['Timestamp'].dt.hour
data['Day'] = data['Timestamp'].dt.day
data['Month'] = data['Timestamp'].dt.month
data['DayOfWeek'] = data['Timestamp'].dt.dayofweek


data['PM2.5_Lag1'] = data['PM2.5'].shift(1)
data['PM2.5_Lag24'] = data['PM2.5'].shift(24)  # daily lag

# Drop NaN values
data = data.dropna()

X = data[['Hour', 'Day', 'Month', 'DayOfWeek', 'PM2.5_Lag1', 'PM2.5_Lag24']]
y = data['PM2.5']

#  training and testing with data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f'MAE: {mae}')
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')

user_input = input("Enter the date for forecasting (YYYY-MM-DD): ")
try:
    target_date = pd.to_datetime(user_input)
except ValueError:
    print("Invalid date format. Please enter the date in YYYY-MM-DD format.")
    exit()

last_known_pm25 = 20
last_known_pm25_lag24 = 40.67

future_data = pd.DataFrame({
    'Timestamp': [target_date],
    'Hour': [0],
    'Day': [target_date.day],
    'Month': [target_date.month],
    'DayOfWeek': [target_date.weekday()],
    'PM2.5_Lag1': [last_known_pm25],
    'PM2.5_Lag24': [last_known_pm25_lag24]
})

future_features = future_data[['Hour', 'Day', 'Month', 'DayOfWeek', 'PM2.5_Lag1', 'PM2.5_Lag24']]
predicted_pm25 = model.predict(future_features)

print(f'The predicted PM2.5 value for {target_date.date()} is {predicted_pm25[0]:.2f}')

plt.figure(figsize=(10, 5))
plt.plot(data['Timestamp'], data['PM2.5'], label='Historical PM2.5')


plt.scatter(future_data['Timestamp'], predicted_pm25, color='red', label='Forecasted PM2.5', zorder=5)

plt.xlabel('Timestamp')
plt.ylabel('PM2.5')
plt.title('PM2.5 Historical and Forecasted Values')
plt.legend()
plt.show()
