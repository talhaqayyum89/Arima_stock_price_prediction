import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import yfinance as yf
import pmdarima as pm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
import streamlit as st
# Set the page layout to wide
st.set_page_config(layout="wide")

warnings.filterwarnings('ignore')

class StockForecasting:
    def __init__(self, symbol, start, end, p_value_threshold=0.05):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.p_value_threshold = p_value_threshold
        self.data = self.download_data()
        self.data = self.sanity_check(self.data)
        self.diff_data = None
        self.best_param = None
        self.model_fit = None

    def download_data(self):
        """Download stock data from Yahoo Finance."""
        data = yf.download(self.symbol, start=self.start, end=self.end, interval='1d')[['Adj Close']]
        return data

    def sanity_check(self, data):
        """Check for missing values and handle them."""
        if data.isnull().values.any():
            print("Missing values found.")
            data.fillna(method='ffill', inplace=True)
            print("Missing values have been forward-filled.")
        else:
            print("No missing values in the data.")
        return data

    def check_stationarity(self, df):
        """Check for stationarity of the data using the ADF test."""
        adf_result = adfuller(df)
        p_value = adf_result[1]
        print(f"ADF Statistic: {adf_result[0]}")
        print(f"p-value: {adf_result[1]}")
        print(f"Critical Values: {adf_result[4]}")
        return p_value < 0.05

    def apply_differencing(self, lag=1):
        """Apply differencing to the series."""
        self.diff_data = self.data['Adj Close'].diff(periods=lag).dropna()
        print(f"Differenced data with lag {lag}:")
        print(self.diff_data.head())
        return self.diff_data

    def plot_acf_pacf(self, diff_data):
        """Plot ACF and PACF for the differenced data."""
        plt.figure(figsize=(14, 7))
        plt.subplot(121)
        plot_acf(diff_data, ax=plt.gca(), lags=40)
        plt.title('Autocorrelation Function')
        plt.subplot(122)
        plot_pacf(diff_data, ax=plt.gca(), lags=40)
        plt.title('Partial Autocorrelation Function')
        plt.tight_layout()
        plt.show()

    def find_best_arima_parameters(self):
        """Find the best ARIMA parameters using auto_arima based on AIC and significant terms."""
        split = int(len(self.data) * 0.8)
        data_train = self.data[:split]
        print("Finding the best ARIMA model...")

        # Use auto_arima to get the initial best model based on AIC
        model = pm.auto_arima(data_train['Adj Close'], seasonal=False, trace=True,
                               error_action='ignore', suppress_warnings=True,
                               stepwise=True, criterion='aic')
        self.best_param = model.order
        print(f"Initial best model based on AIC: {self.best_param}")
        print(model.summary())
        model.plot_diagnostics(figsize=(15,8))


        # Fit the model and check p-values
        arima_model = ARIMA(data_train['Adj Close'], order=self.best_param)
        self.model_fit = arima_model.fit()

        p_values = self.model_fit.pvalues
        print(f"P-values for model terms: {p_values}")

        # Validate model by ensuring no insignificant terms (based on p-value threshold)
        if all(p_values < self.p_value_threshold):
            print(f"Selected model {self.best_param} passes the p-value threshold check.")
        else:
            print(f"Some terms in the model {self.best_param} are not significant.")
            # Use stepwise search to refine
            refined_model = pm.auto_arima(data_train['Adj Close'], seasonal=False,
                                           trace=True, error_action='ignore',
                                           suppress_warnings=True, stepwise=True,
                                           criterion='aic', alpha=self.p_value_threshold)
            refined_model.plot_diagnostics(figsize=(15,8))
            self.best_param = refined_model.order
            self.model_fit = refined_model.fit()
            print(f"Refined model after checking p-values: {self.best_param}")
            print(self.model_fit.summary())

        return self.best_param

    def get_predicted_prices(self, close_prices):
        """Get predicted prices using the best ARIMA model."""
        best_model = ARIMA(close_prices.values, order=self.best_param)
        best_model_fit = best_model.fit(method_kwargs={"warn_convergence": False})
        predictions = best_model_fit.forecast(steps=1)[0]
        return predictions

    def run_forecasting(self):
        """Run the forecasting process and evaluate the strategy."""
        split = int(len(self.data) * 0.8)  # Adjusted split for better training/testing
        data_train = self.data[:split]
        data_test = self.data[split:]

        predictions = []
        for i in range(len(data_test['Adj Close'])):
            current_data = pd.concat([data_train['Adj Close'], data_test['Adj Close'].iloc[:i]])
            next_pred = self.get_predicted_prices(current_data)
            predictions.append(next_pred)

        predictions_df = pd.DataFrame(predictions, columns=['predicted_price'])
        predictions_df.index = data_test.index
        data_test = pd.concat([data_test, predictions_df], axis=1)

        # Generate signals and returns
        data_test['predicted_returns'] = data_test['predicted_price'].pct_change()
        data_test['actual_returns'] = data_test['Adj Close'].pct_change()
        data_test.dropna(inplace=True)
        data_test['signal'] = np.where(data_test['predicted_returns'] >= 0, 1, -1)
        data_test['strategy_returns'] = data_test['signal'] * data_test['actual_returns']
        data_test['cumulative_returns'] = np.cumprod(data_test['strategy_returns'] + 1)

        # Buy and hold returns
        buy_and_hold_returns = (1 + data_test['actual_returns']).cumprod()

        # Error Metrics
        mse = mean_squared_error(data_test['Adj Close'], data_test['predicted_price'])
        mae = mean_absolute_error(data_test['Adj Close'], data_test['predicted_price'])
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((data_test['Adj Close'] - data_test['predicted_price']) / data_test['Adj Close'])) * 100

        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape}%")



        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(data_test['cumulative_returns'], label='Predicted Cumulative Returns')
        plt.plot(buy_and_hold_returns, label='Buy and Hold Cumulative Returns')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.title('Predicted vs Buy and Hold Cumulative Returns')
        plt.legend()
        plt.show()

        return data_test

    def simulate_forecasts(self, num_simulations, forecast_periods):
        """Simulate forecasts for the specified number of periods and simulations."""
        residuals = self.model_fit.resid
        noise_distribution = residuals / np.std(residuals)  # Standardized residuals
        forecast_results = np.zeros((num_simulations, forecast_periods))

        # Simulate forecasts
        for sim in range(num_simulations):
            simulated_noise = np.random.choice(noise_distribution, size=forecast_periods, replace=True)
            forecasted_values = self.model_fit.forecast(steps=forecast_periods) + simulated_noise
            forecast_results[sim, :] = forecasted_values

        return forecast_results

    def percent_simulations_exceeding_value(self, threshold, num_simulations, forecast_periods):
        """Calculate the percentage of simulations exceeding a threshold in each forecast period."""
        forecast_results = self.simulate_forecasts(num_simulations, forecast_periods)
        print(forecast_results)
        # Calculate percentage of simulations exceeding the threshold in each period
        exceed_percentages = (forecast_results > threshold).mean(axis=0) * 100

        # Generate future dates for the forecasts
        split = int(len(self.data) * 0.8)
        data_train = self.data[:split]
        future_dates = pd.date_range(start=data_train.index[-1] + pd.Timedelta(days=1), periods=forecast_periods, freq='B')

        # Create a DataFrame with forecast period and percentage of simulations exceeding threshold
        exceed_df = pd.DataFrame({
            'Date': future_dates,
            'Percentage Exceeding Threshold': exceed_percentages
        })

        return exceed_df

    def plot_simulated_forecasts(self, num_simulations, forecast_periods):
        """Plot simulated forecasts along with actual values."""
        split = int(len(self.data) * 0.8)
        data_train = self.data[:split]
        
        # Get simulated forecasts
        forecast_results = self.simulate_forecasts(num_simulations, forecast_periods)
        
        # Generate future dates for the forecasts
        future_dates = pd.date_range(start=data_train.index[-1] + pd.Timedelta(days=1), periods=forecast_periods, freq='B')

        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 7))
        ax.plot(self.data['Adj Close'], label='Actual Prices', color='blue')
        
        for i in range(num_simulations):
            ax.plot(future_dates, forecast_results[i, :], color='red', alpha=0.1)

        ax.set_title(f'Simulated Forecasts for {self.symbol}')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()

        return fig  # Return the figure

def main():
    st.title("Stock Price Forecasting App")

    # Sidebar for user input
    st.sidebar.header("Input Parameters")

    # Dropdown menu for stock symbols
    stock_symbol = st.sidebar.selectbox(
        'Select Stock Symbol:',
        ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'FB', 'BRK.B', 'NFLX', 'DIS']
    )

    # Date range for analysis
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime('2010-01-01'))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime('2023-12-31'))

    # Input fields for number of simulations, forecast periods, and threshold value
    num_simulations = st.sidebar.number_input("Number of Simulations", min_value=100, max_value=5000, value=1000, step=100)
    forecast_periods = st.sidebar.number_input("Forecast Periods (days)", min_value=1, max_value=365, value=30, step=1)
    threshold_value = st.sidebar.number_input("Threshold Value", min_value=0, value=80, step=1)

    # Initialize StockForecasting class
    st.write(f"Forecasting stock prices for {stock_symbol} from {start_date} to {end_date}...")
    stock_forecast = StockForecasting(symbol=stock_symbol, start=str(start_date), end=str(end_date))

    # Check stationarity
    is_stationary = stock_forecast.check_stationarity(stock_forecast.data['Adj Close'])
    if not is_stationary:
        stock_forecast.apply_differencing(lag=1)

    # Find best ARIMA parameters
    stock_forecast.find_best_arima_parameters()

    # Run forecasting
    data_test = stock_forecast.run_forecasting()

    # Display data_test dataframe on the right
    st.subheader(f"Forecast Results for {stock_symbol}")
    st.write(data_test)

    # Plot results below dataframe
    st.subheader(f"Cumulative Returns Comparison for {stock_symbol}")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(data_test['cumulative_returns'], label='Predicted Cumulative Returns', color='green')
    ax.plot((1 + data_test['actual_returns']).cumprod(), label='Buy and Hold Cumulative Returns', color='blue')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Returns')
    ax.set_title('Predicted vs Buy and Hold Cumulative Returns')
    ax.legend()
    st.pyplot(fig)

    # Plot simulated forecasts
    st.subheader(f"Simulated Forecasts for {stock_symbol}")
    simulated_forecast_fig = stock_forecast.plot_simulated_forecasts(num_simulations=num_simulations, forecast_periods=forecast_periods)
    st.pyplot(simulated_forecast_fig)  # Render the returned figure

    # Show percentage of simulations exceeding threshold
    st.subheader(f"Percentage of Simulations Exceeding ${threshold_value} in Forecast Periods")
    exceed_df = stock_forecast.percent_simulations_exceeding_value(threshold_value, num_simulations, forecast_periods)
    st.write(exceed_df)

if __name__ == "__main__":
    main()
