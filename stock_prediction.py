import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# Function to download stock data and predict prices using LSTM
def stock_prediction():
    start_date = str(input("Enter the Start date for the dataset (format: yyyy-mm-dd): "))
    end_date = str(input("Enter the End date for the dataset (format: yyyy-mm-dd): "))
    symbol = str(input("Enter the ticker symbol: "))
    
    # Download stock data
    df = yf.download(symbol, start_date, end_date)

    # Display the data
    plt.figure(figsize=(16, 8))
    plt.plot(df.index, df['Close'], label='Close Price')
    plt.title(f"{symbol} Stock Price from {start_date} to {end_date}")
    plt.xlabel('Date')
    plt.ylabel('Price (INR)')
    plt.legend()
    plt.show()

    # Data preprocessing
    df = df.reset_index()
    data = df.sort_index(ascending=True, axis=0)

    new_data = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])
    for i in range(0, len(data)):
        new_data['Date'][i] = data['Date'][i]
        new_data['Close'][i] = data['Close'][i]
    new_data.index = new_data.Date
    new_data.drop('Date', axis=1, inplace=True)

    dataset = new_data.values

    train_size = int(len(dataset) * 0.8)
    train, valid = dataset[0:train_size, :], dataset[train_size:, :]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    x_train, y_train = [], []
    for i in range(60, len(train)):
        x_train.append(scaled_data[i - 60:i, 0])
        y_train.append(scaled_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # Build LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x_train, y_train, epochs=10, batch_size=1, verbose=2)

    # Make predictions
    inputs = new_data[len(new_data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)

    X_test = []
    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i - 60:i, 0])
    X_test = np.array(X_test)

    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)

    valid_predicted = closing_price[-len(valid):]

    # Evaluate and visualize results
    rms = np.sqrt(np.mean(np.power((valid - closing_price), 2)))
    print("The root mean square error is ", rms)

    train = new_data[:train_size]
    valid = new_data[train_size:]
    valid['Predictions'] = valid_predicted

    plt.figure(figsize=(16, 8))
    plt.title('LSTM Model')
    plt.xlabel('Date')
    plt.ylabel('Close Price (INR)')
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()

    plt.figure(figsize=(16, 8))
    plt.title('LSTM Model')
    plt.xlabel('Date')
    plt.ylabel('Close Price (INR)')
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Val', 'Predictions'], loc='lower right')
    plt.show()

if __name__ == "__main__":
    stock_prediction()
