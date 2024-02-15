import yfinance as yf
import os

def fetch_stock_data(stock_symbol, period):
    """
    Pobiera i zwraca dane o akcjach dla danego symbolu i okresu.
    - stock_symbol: symbol giełdowy akcji do pobrania (np. "AAPL" dla Apple Inc.)
    - period: okres czasu, dla którego mają być pobrane dane (np. "1mo" dla jednego miesiąca)
    """
    ticker = yf.Ticker(stock_symbol)
    history = ticker.history(period=period)
    history.reset_index(inplace=True)  # Resetowanie indeksu, aby uzyskać kolumnę 'Date'
    history['Date'] = history['Date'].dt.date  # Konwersja kolumny datetime na datę, usuwając czas
    return history

def main():
    """
    Główna funkcja wykonująca pobieranie i wyświetlanie danych dla akcji.
    """
    # Definiowanie list akcji do śledzenia
    stocks = [
"AAPL", "MSFT", "AMZN", "GOOGL", "TSLA",  "V", "JNJ", "WMT", "JPM", "UNH", "PG", "NVDA", "HD", "DIS", "MA", "PYPL", "BABA", "VOO", "INTC", "ADBE", "CRM", "NFLX", "ORCL", "T", "KO", "PEP", "NKE", "XOM", "CVX", "BA", "MMM", "MCD", "IBM", "GS", "CSCO", "MRK", "BAC", "GE"]
    data_period_days = "65d"  # Przygotowanie formatu okresu dla akcji

    # Tworzenie folderów, jeśli nie istnieją
    os.makedirs("stocks", exist_ok=True)

    print("Dane dla akcji:")
    for stock in stocks:
        history = fetch_stock_data(stock, data_period_days)
        history.to_csv(f"stocks/{stock}_stock_data.csv", index=False)  # Zapisywanie danych do pliku CSV w folderze 'stocks'
        print(f"Dane dla {stock}:")
        print(history.head())  # Wyświetlanie 5 rekordów dla uproszczenia
        print("\n")

