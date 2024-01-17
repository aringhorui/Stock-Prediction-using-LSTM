# Stock Prediction using LSTM

This project employs Long Short-Term Memory (LSTM) neural networks to forecast stock prices based on historical data. The Python script downloads historical stock price data using the `yfinance` library, preprocesses the data, and builds an LSTM model using the Keras library. The trained model is then used to make predictions on future stock prices.

## Table of Contents

- [Key Features](#key-features)
- [How to Use](#how-to-use)
- [Prerequisites](#prerequisites)
- [Installation](#installation)


## Key Features

- **Data Collection:** Utilizes the `yfinance` library to fetch historical stock prices for a given time period and ticker symbol.

- **Data Preprocessing:** Prepares the data by normalizing it using Min-Max scaling and structuring it into sequences suitable for LSTM input.

- **LSTM Model Training:** Builds an LSTM neural network using Keras with two LSTM layers and a dense layer for prediction. The model is trained on historical stock price data.

- **Prediction and Visualization:** Makes predictions on future stock prices using the trained model and visualizes the results using matplotlib.

## How to Use

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/aringhorui/Stock-Prediction-using-LSTM.git
2. **Navigate to the Project Directory:**
   ```bash
   cd Stock-Prediction-using-LST
3. **run the Script:**
   ```bash
   python stock_prediction.py  
## Prerequisites

1.**Python 3**

2.**Libraries:** yfinance, pandas, numpy, matplotlib, scikit-learn, keras

## Installation
```bash
pip install yfinance pandas numpy matplotlib scikit-learn keras
