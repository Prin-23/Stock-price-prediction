import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

# Set time range
start = '2010-01-01'
end = '2019-12-31'

# App title
st.title('📈 Stock Trend Prediction')

# User input
user_input = st.text_input('Enter Stock Ticker', 'AAPL')
st.caption("Try: AAPL, TSLA, AMZN, MSFT, INFY.NS")

# Show quick links to Yahoo & Google Finance
yahoo_url = f"https://finance.yahoo.com/quote/{user_input}"
google_url = f"https://www.google.com/finance/quote/{user_input}"
st.markdown(f"🔗 [Yahoo Finance - {user_input}]({yahoo_url})")
st.markdown(f"🔗 [Google Finance - {user_input}]({google_url})")

# Cached data loader to avoid rate-limiting
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

if user_input:
    df = load_data(user_input, start, end)

    if df.empty:
        st.error("❌ No data found for this ticker. Please try a different one.")
    else:
        # Show data summary
        st.subheader('Data from 2010 - 2019')
        st.write(df.describe())

        # Plot closing price
        st.subheader('Closing Price vs Time chart')
        fig1 = plt.figure(figsize=(12, 6))
        plt.plot(df['Close'])
        st.pyplot(fig1)

        # Plot with 100MA
        st.subheader('Closing Price vs Time chart with 100MA')
        ma100 = df['Close'].rolling(100).mean()
        fig2 = plt.figure(figsize=(12, 6))
        plt.plot(df['Close'], label='Close')
        plt.plot(ma100, label='100MA')
        plt.legend()
        st.pyplot(fig2)

        # Plot with 100MA and 200MA
        st.subheader('Closing Price vs Time chart with 100MA & 200MA')
        ma200 = df['Close'].rolling(200).mean()
        fig3 = plt.figure(figsize=(12, 6))
        plt.plot(df['Close'], label='Close')
        plt.plot(ma100, label='100MA')
        plt.plot(ma200, label='200MA')
        plt.legend()
        st.pyplot(fig3)

        # Split data
        data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
        data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

        if len(data_training) >= 100 and len(data_testing) > 0:
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_training_array = scaler.fit_transform(data_training)

            # Load model
            try:
                model = load_model('Latest_stock_price_model.keras')
            except OSError:
                st.error("❌ Model file not found. Please make sure 'Latest_stock_price_model.keras' exists in the same directory.")
                st.stop()

            # Prepare input for test
            past_100_days = data_training.tail(100)
            final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
            input_data = scaler.fit_transform(final_df)

            x_test = []
            y_test = []

            for i in range(100, input_data.shape[0]):
                x_test.append(input_data[i - 100:i])
                y_test.append(input_data[i, 0])

            if len(x_test) > 0:
                x_test, y_test = np.array(x_test), np.array(y_test)

                # Predict
                y_predicted = model.predict(x_test)

                # Reverse scale
                scale_factor = 1 / scaler.scale_[0]
                y_predicted = y_predicted * scale_factor
                y_test = y_test * scale_factor

                # Plot prediction
                st.subheader('📊 Predictions vs Original')
                fig4 = plt.figure(figsize=(12, 6))
                plt.plot(y_test, 'b', label='Original Price')
                plt.plot(y_predicted, 'r', label='Predicted Price')
                plt.xlabel('Time')
                plt.ylabel('Price')
                plt.legend()
                st.pyplot(fig4)
            else:
                st.warning("⚠️ Not enough test data to make predictions.")
        else:
            st.warning("⚠️ Not enough training data. Try a stock with more historical data.")
