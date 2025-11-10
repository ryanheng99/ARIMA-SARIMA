## ARIMA-SARIMA
## 📈 Sensor Data Forecasting with ARIMA & SARIMA
This repository contains a time series forecasting project using ARIMA and SARIMA models applied to sensor data from a manufacturing process. The goal is to evaluate model performance and identify the most suitable forecasting approach for industrial sensor signals.
🧠 Models Used


ARIMA (AutoRegressive Integrated Moving Average)
Captures linear trends and autocorrelation in non-seasonal time series.


SARIMA (Seasonal ARIMA)
Extends ARIMA to handle seasonality in time series data.


## 📁 Contents

## SensordataForecastwith_ARIMA.ipynb
```
Implements ARIMA modeling, including:

Stationarity checks
Parameter tuning
Forecasting and evaluation
```


## SensordataForecastwith_SARIMA.ipynb
```
Implements SARIMA modeling for seasonal data, including:

Seasonal decomposition
Model fitting
Forecast visualization
```


## 📊 Dataset
The dataset used simulates sensor readings from a manufacturing environment. It includes timestamped values that exhibit both trend and seasonal patterns, making it suitable for SARIMA modeling.
🛠️ Requirements
Install dependencies using:
Shellpip install pandas numpy matplotlib statsmodelsShow more lines
## 🚀 How to Run


Clone the repository:
```
git clone https://github.com/ryanheng99/ARIMA-SARIMA.git
cd ARIMA-SARIMA
```

Open the notebook:
```
jupyter notebook SensordataForecastwith_SARIMA.ipynb
```

Run the cells to explore the forecasting workflow.


📌 Notes

Ensure the time series is indexed by datetime for proper modeling.
SARIMA parameters are selected based on ACF/PACF plots and seasonal decomposition.
