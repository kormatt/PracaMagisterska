import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from alpaca_trade_api.rest import REST, TimeFrame
import CheckPrices
import matplotlib.pyplot as plt

# Ustawienie kluczy API Alpaca
api_key = '####################'
api_secret = '##############################'
api = REST(api_key, api_secret, base_url='https://paper-api.alpaca.markets')


def calculate_sma(data, window):
    return data.rolling(window=window).mean()

def calculate_rsi(data, window=14):
    delta = data.diff(1)
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Funkcja do ładowania danych historycznych dla danej akcji
def load_stock_data(stock_file):
    """
    Funkcja wczytuje dane historyczne dla danej akcji z pliku CSV.

    Parametry:
    - stock_file: Ścieżka do pliku CSV z danymi historycznymi akcji.

    Zwraca:
    - data: DataFrame zawierający dane historyczne dla danej akcji.
    """
    all_data = []
    data = pd.read_csv(stock_file)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data['SMA_30'] = calculate_sma(data['Close'], 30)
    data['RSI_14'] = calculate_rsi(data['Close'], 14)
    all_data.append(data)
    return pd.concat(all_data, axis=0).dropna()

# Funkcja do przetwarzania danych i przygotowania do treningu
def preprocess_data(data):
    """
    Funkcja przetwarza dane historyczne akcji i przygotowuje je do treningu modelu.

    Parametry:
    - data: DataFrame zawierający dane historyczne dla danej akcji.

    Zwraca:
    - X: Tablica zawierająca dane wejściowe modelu.
    - y: Tablica zawierająca dane wyjściowe modelu.
    - scaler: Obiekt MinMaxScaler użyty do skalowania danych.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    X, y = [], []
    time_steps = 60
    for i in range(time_steps, len(scaled_data)):
        X.append(scaled_data[i-time_steps:i, 0])
        y.append(scaled_data[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    
    return X, y, scaler

# Funkcja do budowania i trenowania modelu LSTM
def train_model(X, y, epochs):
    """
    Funkcja buduje i trenuje model LSTM na podstawie danych wejściowych i wyjściowych.

    Parametry:
    - X: Tablica danych wejściowych modelu.
    - y: Tablica danych wyjściowych modelu.
    - epochs: Liczba epok treningowych.

    Zwraca:
    - model: Wytrenowany model LSTM.
    """
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
        LSTM(units=50),
        Dense(units=25),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=epochs, batch_size=32)
    return model

def make_decision(last_real_price, predicted_price):
    """
    Funkcja podejmuje decyzję inwestycyjną na podstawie różnicy między rzeczywistą ceną a przewidywaną ceną.

    Parametry:
    - last_real_price: Ostatnia rzeczywista cena akcji.
    - predicted_price: Przewidziana cena akcji.

    Zwraca:
    - decision: Decyzja inwestycyjna ('Kup', 'Sprzedaj', 'Trzymaj').
    """
    threshold_buy = 0.02
    threshold_sell = -0.02
    change_percent = (predicted_price - last_real_price) / last_real_price
    if change_percent > threshold_buy:
        return 'Kup'
    elif change_percent < threshold_sell:
        return 'Sprzedaj'
    else:
        return 'Trzymaj'

def make_decision_and_trade(symbol, model, scaler, data, account):
    """
    Funkcja podejmuje decyzję inwestycyjną dla danej akcji i ewentualnie dokonuje zakupu/sprzedaży.

    Parametry:
    - symbol: Symbol danej akcji.
    - model: Wytrenowany model do przewidywania cen.
    - scaler: Obiekt MinMaxScaler użyty do skalowania danych.
    - data: Dane historyczne dla danej akcji.
    - account: Informacje o koncie handlowym.

    Zwraca:
    - None
    """
    last_60_days = data['Close'][-60:].values.reshape(-1, 1)  # Ostatnie 60 dni cen
    last_60_days_scaled = scaler.transform(last_60_days)
    
    X_test = np.array([last_60_days_scaled])
    X_test = X_test.reshape((1, X_test.shape[1], 1))  # Poprawka kształtu danych wejściowych
    
    predicted_price = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price)

    # Pobieranie ostatniej ceny zamknięcia
    barset = api.get_bars(symbol, TimeFrame.Day, limit=1).df
    last_close = barset['close'].iloc[-1]
    
    print(f"{symbol} - Ostatnia cena zamknięcia: {last_close}, Przewidywana cena: {predicted_price[0][0]}")
    decision = make_decision(last_close, predicted_price[0][0])
    print(f"{symbol} - Decyzja: {decision}")
    with open('logi.txt', 'a') as file:
        file.write(f"\n{symbol} - Ostatnia cena zamkniecia: {last_close}, Przewidywana cena: {predicted_price[0][0]} - Decyzja: {decision} ",)
    # Obliczanie dostępnego kapitału
    total_capital = float(account.cash)
    investment_per_stock_percentage = 0.05  # Procent kapitału do zainwestowania na akcję
    last_close_price = barset['close'].iloc[-1]
    investment_per_stock = total_capital * investment_per_stock_percentage
    shares_to_buy = investment_per_stock // last_close_price

    # Wykonanie odpowiedniego zlecenia w zależności od decyzji
    try:
        if decision == 'Kup':
            print(f"Składanie zlecenia kupna dla {symbol}")
            api.submit_order(symbol=symbol, qty=shares_to_buy, side='buy', type='market', time_in_force='gtc')
        elif decision == 'Sprzedaj':
            print(f"Składanie zlecenia sprzedaży dla {symbol}")
            try:
                position = api.get_position(symbol)
                if position:
                    api.submit_order(symbol=symbol, qty=1, side='sell', type='market', time_in_force='gtc')
            except Exception as e:
                print(f"Nie można złożyć zlecenia sprzedaży, nie posiadasz {symbol}: {e}")
    except Exception as e:
        print(f"Wystąpił błąd podczas próby handlu {symbol}: {e}")

def main():
    """
    Główna funkcja programu.
    """
    CheckPrices.main()

    account = api.get_account()  # Pobieranie informacji o koncie
    epochs = 70  # Liczba epok do treningu modelu
    folder_path = "stocks"

    # Pętla po plikach CSV w folderze stocks
    for filename in os.listdir(folder_path):
        if filename.endswith("_stock_data.csv"):
            symbol = filename.split('_')[0]
            print(f"Przetwarzanie {symbol}")
            
            data = load_stock_data(os.path.join(folder_path, filename))
            X, y, scaler = preprocess_data(data)
            model = train_model(X, y, epochs)
            make_decision_and_trade(symbol, model, scaler, data, account)

if __name__ == "__main__":
    main()
