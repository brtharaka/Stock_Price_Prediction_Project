# Stock Price Prediction Project 📈  

[![Python](https://img.shields.io/badge/Python-3.9-blue)](https://www.python.org/)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0-orange)](https://www.tensorflow.org/)  
[![Flask](https://img.shields.io/badge/Flask-WebApp-green)](https://flask.palletsprojects.com/)  
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  

> 🚀 **Stock Price Forecasting with GRU (best model), LSTM, and ARIMA — deployed with Flask + HTML/CSS.**


This project uses machine learning models to predict stock prices based on historical data.  
We trained multiple models (LSTM, GRU, and ARIMA) and evaluated them using RMSE, MAE, and MAPE.  
✅ Based on the results, the **GRU model** achieved the best accuracy.

---

## 🧠 Technologies Used
- Python
- Jupyter Notebook / Google Colab
- Pandas, Numpy
- Scikit-learn
- Matplotlib / Seaborn
- TensorFlow / Keras
- Flask (for deployment)
- HTML / CSS (for interface)

---

## 📊 Dataset

This project uses a historical stock dataset (`stock_data.csv`), which includes daily stock market data for Apple Inc. (AAPL).

| Column Name | Description |
|-------------|-------------|
| **Date**      | The date of the trading session (YYYY-MM-DD) |
| **Open**      | Stock price at market opening |
| **High**      | Highest price reached during session |
| **Low**       | Lowest price reached during session |
| **Close**     | Stock price at market close |
| **Adj Close** | Closing price adjusted for dividends/splits |
| **Volume**    | Total number of shares traded |

This dataset is ideal for:
- 📈 Time series forecasting  
- 🧠 ML model training  
- 🔍 Trend and pattern detection  

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/stock-price-prediction.git
cd stock-price-prediction
