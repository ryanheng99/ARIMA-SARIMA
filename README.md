# ARIMA-SARIMA Forecasting for Manufacturing

A time series forecasting project that evaluates and compares ARIMA and SARIMA models on manufacturing industry datasets to identify the most suitable forecasting approach.

## 🎯 Overview

This project analyzes manufacturing time series data to determine whether ARIMA (AutoRegressive Integrated Moving Average) or SARIMA (Seasonal ARIMA) provides better forecasting performance. It includes model testing, parameter optimization, and performance comparison for production planning and demand forecasting.

## 🏭 Use Cases

- **Production Planning** - Forecast manufacturing output and capacity needs
- **Demand Forecasting** - Predict product demand patterns
- **Inventory Management** - Optimize stock levels based on forecasts
- **Quality Control** - Predict parameter trends in manufacturing processes
- **Resource Allocation** - Plan workforce and material requirements

## ✨ Features

- **Dual Model Testing** - Compare ARIMA and SARIMA side-by-side
- **Automated Model Selection** - Find optimal parameters using statistical tests
- **Seasonality Detection** - Identify and model seasonal patterns
- **Performance Metrics** - Evaluate models using RMSE, MAE, MAPE
- **Visualization** - Plot forecasts, residuals, and diagnostic checks
- **Manufacturing Focus** - Tailored for industrial time series characteristics

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.7+
pandas
numpy
statsmodels
matplotlib
seaborn
scikit-learn
```

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ryanheng99/ARIMA-SARIMA.git
   cd ARIMA-SARIMA
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   
   Or install manually:
   ```bash
   pip install pandas numpy statsmodels matplotlib seaborn scikit-learn pmdarima
   ```

3. **Prepare your data:**
   - Place your manufacturing dataset in the project directory
   - Ensure time series data has a datetime index
   - Format: CSV with timestamp and target variable columns

## 📊 Methodology

### ARIMA Model

**Components:**
- **AR (p)** - AutoRegressive: Uses past values to predict future
- **I (d)** - Integrated: Number of differencing operations for stationarity
- **MA (q)** - Moving Average: Uses past forecast errors

**Best for:** Non-seasonal data, trend-based manufacturing metrics

### SARIMA Model

**Additional Components:**
- **Seasonal AR (P)** - Seasonal autoregressive component
- **Seasonal I (D)** - Seasonal differencing
- **Seasonal MA (Q)** - Seasonal moving average
- **m** - Seasonal period (e.g., 12 for monthly, 7 for daily)

**Best for:** Data with clear seasonal patterns (e.g., weekly production cycles, monthly demand)

## 📖 Usage

### Basic Workflow

```python
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load manufacturing data
df = pd.read_csv('manufacturing_data.csv', parse_dates=['date'], index_col='date')

# Check for stationarity
from statsmodels.tsa.stattools import adfuller
result = adfuller(df['production_volume'])
print(f'ADF Statistic: {result[0]}')
print(f'p-value: {result[1]}')

# Fit ARIMA model
arima_model = ARIMA(df['production_volume'], order=(1, 1, 1))
arima_result = arima_model.fit()

# Fit SARIMA model
sarima_model = SARIMAX(df['production_volume'], 
                        order=(1, 1, 1), 
                        seasonal_order=(1, 1, 1, 12))
sarima_result = sarima_model.fit()

# Generate forecasts
arima_forecast = arima_result.forecast(steps=30)
sarima_forecast = sarima_result.forecast(steps=30)

# Compare performance
from sklearn.metrics import mean_squared_error, mean_absolute_error
print(f'ARIMA RMSE: {mean_squared_error(test_data, arima_pred, squared=False)}')
print(f'SARIMA RMSE: {mean_squared_error(test_data, sarima_pred, squared=False)}')
```

### Model Selection Process

```
┌─────────────────────────────────────────────────────────────┐
│                   DATA PREPARATION                          │
├─────────────────────────────────────────────────────────────┤
│  • Load manufacturing time series data                      │
│  • Handle missing values and outliers                       │
│  • Split into train/test sets (80/20 or 70/30)            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              EXPLORATORY DATA ANALYSIS                      │
├─────────────────────────────────────────────────────────────┤
│  • Visualize time series patterns                          │
│  • Test for stationarity (ADF test)                        │
│  • Identify seasonality (ACF/PACF plots)                   │
│  • Determine differencing needed                           │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                  MODEL FITTING                              │
├─────────────────────────────────────────────────────────────┤
│  ARIMA:                      SARIMA:                        │
│  • Grid search for (p,d,q)  • Grid search for (p,d,q)(P,D,Q,m) │
│  • Fit on training data     • Fit on training data         │
│  • Check residuals          • Check residuals              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              EVALUATION & COMPARISON                        │
├─────────────────────────────────────────────────────────────┤
│  • Calculate metrics: RMSE, MAE, MAPE                      │
│  • Perform residual diagnostics                            │
│  • Compare AIC/BIC scores                                  │
│  • Select best performing model                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    FORECASTING                              │
├─────────────────────────────────────────────────────────────┤
│  • Generate predictions on test set                        │
│  • Produce future forecasts with confidence intervals      │
│  • Visualize results                                       │
└─────────────────────────────────────────────────────────────┘
```

## 📈 Performance Metrics

### Evaluation Criteria

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **RMSE** | √(Σ(y-ŷ)²/n) | Lower is better - penalizes large errors |
| **MAE** | Σ\|y-ŷ\|/n | Lower is better - average absolute error |
| **MAPE** | Σ\|y-ŷ\|/y × 100 | Lower is better - percentage error |
| **AIC** | Model statistic | Lower is better - penalizes complexity |
| **BIC** | Model statistic | Lower is better - stronger penalty for complexity |

### Model Selection Guidelines

**Choose ARIMA when:**
- No clear seasonal pattern exists
- Data shows trend but not cyclical behavior
- Simpler model performs adequately
- Computational efficiency is important

**Choose SARIMA when:**
- Clear seasonal patterns detected (weekly, monthly, quarterly cycles)
- Manufacturing has cyclical demand or production schedules
- Improved accuracy justifies additional complexity
- Seasonal decomposition shows significant seasonal component

## 🔍 Key Statistical Tests

### Stationarity Testing
```python
# Augmented Dickey-Fuller test
from statsmodels.tsa.stattools import adfuller

def check_stationarity(timeseries):
    result = adfuller(timeseries)
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    if result[1] <= 0.05:
        print("Data is stationary")
    else:
        print("Data is non-stationary - differencing required")
```

### Seasonality Detection
```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose time series
decomposition = seasonal_decompose(df['production_volume'], 
                                   model='additive', 
                                   period=12)
decomposition.plot()
```

## 📁 Project Structure

```
ARIMA-SARIMA/
├── data/
│   ├── raw/                    # Original manufacturing datasets
│   └── processed/              # Cleaned and preprocessed data
├── notebooks/
│   ├── 01_EDA.ipynb           # Exploratory data analysis
│   ├── 02_ARIMA.ipynb         # ARIMA model development
│   ├── 03_SARIMA.ipynb        # SARIMA model development
│   └── 04_Comparison.ipynb    # Model comparison
├── src/
│   ├── preprocessing.py        # Data cleaning functions
│   ├── model_selection.py      # Grid search and parameter tuning
│   └── evaluation.py           # Performance metrics
├── results/
│   ├── plots/                  # Visualization outputs
│   └── forecasts/              # Prediction results
├── requirements.txt
└── README.md
```

## 🛠️ Hyperparameter Tuning

### ARIMA Parameters

- **p (AR order):** Number of lag observations (typically 0-5)
- **d (Differencing):** Number of times to difference (0-2)
- **q (MA order):** Size of moving average window (typically 0-5)

### SARIMA Additional Parameters

- **P (Seasonal AR):** Seasonal autoregressive order (0-2)
- **D (Seasonal Differencing):** Seasonal differencing (0-1)
- **Q (Seasonal MA):** Seasonal moving average order (0-2)
- **m (Season Length):** Period of seasonality (7, 12, 24, etc.)

### Auto-Selection with Auto ARIMA
```python
from pmdarima import auto_arima

# Automatically find best parameters
auto_model = auto_arima(df['production_volume'],
                        seasonal=True,
                        m=12,  # monthly seasonality
                        stepwise=True,
                        trace=True)
print(auto_model.summary())
```

## 📊 Visualization Examples

- **Time Series Plot** - Original data with trend and seasonality
- **ACF/PACF Plots** - Identify AR and MA orders
- **Residual Diagnostics** - Check for white noise in residuals
- **Forecast Plot** - Predictions with confidence intervals
- **Model Comparison** - Side-by-side performance visualization

## 🎓 Learning Resources

- [ARIMA Documentation - Statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html)
- [SARIMA Guide - Statsmodels](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)
- [Time Series Analysis with Python](https://otexts.com/fpp3/)

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional forecasting models (Prophet, LSTM)
- More manufacturing datasets
- Advanced feature engineering
- Automated reporting dashboard
- Model deployment pipeline

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👤 Author

**Ryan Heng**
- GitHub: [@ryanheng99](https://github.com/ryanheng99)
- Focus: Time series forecasting and manufacturing analytics

## 🙏 Acknowledgments

- Built for manufacturing forecasting applications
- Utilizes statsmodels for robust time series analysis
- Inspired by real-world production planning challenges

---

⭐ **Found this useful?** Star this repo to support time series forecasting in manufacturing!

📫 **Questions?** Open an issue or reach out via GitHub.
