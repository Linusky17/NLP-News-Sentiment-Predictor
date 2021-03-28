import pandas as pd
import streamlit as st
from pandas_datareader import data
from datetime import datetime, timedelta
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np

st.set_page_config(page_title='Stock Predictor', page_icon = 'icon.png')
st.title("Stock Price Predictor")
stock_name = st.text_input("Enter the stock name: \n", "AAPL")
st.header('View Stock History')
option = st.slider("How many days of data would you like to see?", 30, 180, 90)
end = datetime.today().strftime('%Y-%m-%d')
start = (datetime.today() - timedelta(option)).strftime('%Y-%m-%d')

@st.cache
def load_data(stock, start_date, end_date):
    if stock == "":
        stock = "AAPL"
    start_date = '2015-1-1'
    end_date = '2019-12-31'
    df = data.DataReader(name=stock, start=start_date, end=end_date, data_source='yahoo')
    return df


data_load_state = st.text("Loading data...")

df = load_data(stock=stock_name, start_date=start, end_date=end)
df.sort_index(axis=0, inplace=True, ascending=False)

st.subheader(f'{stock_name} Stock Prices for the Past {option} Days')
st.dataframe(df)

chart_data = df[['Close']]
st.subheader("Prices at Close")
st.line_chart(chart_data)

data_load_state.text("Data loaded!")


st.header('Stock Price Predictor')
hist = []
target = []
length = option
adj_close = df['Adj Close']

#st.text(f'{len}')


for i in range(len(adj_close) - length):
   x = adj_close[i:i+length]
   y = adj_close[i+length]
   hist.append(x)
   target.append(y)

#st.text(f'{target}')
hist = np.array(hist)
target = np.array(target)
target = target.reshape(-1,1)


st.text(f'{hist.shape}')


#train/test split
X_train = hist[:1138]
X_test = hist[1138:]
y_train = target[:1138]
y_test = target[1138:]

sc = MinMaxScaler()
#train set, fit_transform
X_train_scaled = sc.fit_transform(X_train)
y_train_scaled = sc.fit_transform(y_train)
#test set, only transform
X_test_scaled = sc.transform(X_test)
y_test_scaled = sc.transform(y_test)

X_train_scaled = X_train_scaled.reshape((len(X_train_scaled), length, 1))
X_test_scaled = X_test_scaled.reshape((len(X_test_scaled), length, 1))
