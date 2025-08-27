from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import datetime
import os

app = Flask(__name__)

# Load the GRU model with error handling
try:
    model = load_model("models/gru_model (1).h5")  # <-- fixed filename
except Exception as e:
    model = None
    print(f"Error loading model: {e}")

def get_stock_data(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end)
        return df[['Close']]
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def predict_prices(df, seq_len=60):
    if model is None:
        raise ValueError("Model is not loaded.")
    if df.empty or len(df) < seq_len:
        raise ValueError("Not enough data for prediction.")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    last_seq = scaled_data[-seq_len:]
    X_input = np.reshape(last_seq, (1, seq_len, 1))
    prediction = model.predict(X_input)
    prediction = scaler.inverse_transform(prediction)
    return float(prediction[0][0])

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    message = None
    if request.method == "POST":
        start_date = request.form["start"]
        end_date = request.form["end"]
        df = get_stock_data("AAPL", start=start_date, end=end_date)
        try:
            prediction = predict_prices(df)
        except ValueError as ve:
            message = str(ve)
        except Exception as e:
            message = f"Unexpected error: {e}"
    return render_template("index.html", prediction=prediction, message=message)

if __name__ == "__main__":
    app.run(debug=True)