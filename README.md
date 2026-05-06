# 📈 Time Series Forecasting — Airline Passenger Demand

An end-to-end time series project comparing classical and modern
forecasting methods on airline passenger data.

## 🔗 Live Demo
[Click here to view the live dashboard](https://time-series-forecasting-wxcatde8i66yaanjgawkpy.streamlit.app/)

## 📌 Project Overview
This project demonstrates four different forecasting approaches:
- **ARIMA** — Classical autoregressive model
- **SARIMA** — ARIMA extended with seasonal components
- **Facebook Prophet** — Modern trend-aware forecasting
- **LSTM** — Deep learning sequence model

Each model is evaluated on the same test period and compared
using MAE, RMSE, and MAPE metrics.

## 📊 Key Findings
- SARIMA consistently outperforms ARIMA due to strong seasonality
- Prophet provides the most business-friendly forecasts
- LSTM shows competitive performance but requires more data
- All models correctly identify the July/August seasonal peak

## 🛠️ Tools & Technologies
| Area | Tools |
|---|---|
| Language | Python 3.11 |
| Classical Models | Statsmodels (ARIMA, SARIMA) |
| Modern Models | Facebook Prophet, TensorFlow/Keras (LSTM) |
| Visualisation | Plotly, Matplotlib |
| Dashboard | Streamlit |
| Evaluation | Scikit-learn, NumPy |
| IDE | VS Code |

## 🚀 How to Run Locally
```bash
git clone https://github.com/YOUR_USERNAME/time-series-forecasting.git
cd time-series-forecasting
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure
```
time-series-forecasting/
├── Data/               # Air passengers dataset
├── Notebook/           # EDA and analysis notebook
├── model_results/      # Saved forecasts and metrics
├── app.py              # Streamlit dashboard
├── forecasting.py      # All model functions
├── requirements.txt
└── README.md
```

## 💼 Business Impact
Accurate demand forecasting enables airlines, retailers, and
manufacturers to optimise staffing, inventory, and capacity
planning — directly reducing costs and improving service levels.
