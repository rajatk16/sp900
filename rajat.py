#!/usr/local/bin/python3
#!coding utf-8

# import libraries
import bs4 as bs
import requests
import csv
import os
import math
import pandas_datareader.data as web
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

style.use('ggplot')

# Get SP500 tickers
def save_sp500_tickers():
  resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
  soup = bs.BeautifulSoup(resp.text, 'lxml')
  table = soup.find('table', {'class': 'wikitable sortable'})
  tickers = []
  for row in table.findAll('tr')[1:]:
    ticker = row.findAll('td')[0].text
    ticker = ticker.strip('\n')
    tickers.append(ticker)
  return tickers

# # Get SP400 tickers
def save_sp400_tickers():
  resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_400_companies')
  soup = bs.BeautifulSoup(resp.text, 'lxml')
  table = soup.find('table', {'class': 'wikitable sortable'})
  tickers = []
  for row in table.findAll('tr')[1:]:
    ticker = row.findAll('td')[1].text
    ticker = ticker.strip('\n')
    tickers.append(ticker)
  return tickers

# Save SP500 closing prices in a csv
def get_data_from_yahoo():
  main_df = pd.DataFrame()
  tickers = set(save_sp500_tickers() + save_sp400_tickers())
  tickers = list(tickers)
  start = dt.datetime(2010,1,1)
  end = dt.datetime.now()
  for ticker in tickers:
    try:
      df = web.DataReader(ticker, 'yahoo', start, end)
      df.rename(columns={'Adj Close': ticker}, inplace=True)
      df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)
      if main_df.empty:
        main_df = df
      else:
        main_df = main_df.join(df)
    except KeyError:
      pass
  main_df.to_csv('sp900.csv')

def data_preprocessing():
  df = pd.read_csv('sp900.csv', index_col=0)
  tickers = df.columns.values.tolist()
  for ticker in tickers:
    if(df[ticker].isnull().sum() > 0):
      print('{} has empty values'.format(ticker))
      df[ticker].fillna(0, inplace=True)
  df.to_csv('sp900.csv')

def arima(ticker):
  # Get the data from CSV
  df = pd.read_csv('sp900.csv')
  # Take the data for the ticker from the CSV
  series = df[ticker].values
  X = series
  fullerresult = adfuller(X)
  print("ADF Statistics: %f" % fullerresult[0])
  print("P value %f" % fullerresult[1])
  print("Critical values: ")
  for key,value in fullerresult[4].items():
    print('\t%s: %.3f' % (key, value))
  size = int(len(X) * 0.66)
  train, test = X[0:size], X[size:len(X)]
  history = [x for x in train]
  predictions = list()
  with open('{}.csv'.format(ticker), 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)  
    filewriter.writerow(['Predicted', 'Expected'])
    for t in range(len(test)):
      model = ARIMA(history, order=(5,1,0))
      model_fit = model.fit(disp=0)
      output = model_fit.forecast()
      yhat = output[0]
      predictions.append(yhat)
      obs = test[t]
      history.append(obs)
      filewriter.writerow(['%.3f' % (yhat), '%.3f' % (obs)])
  rmse = math.sqrt(mean_squared_error(test, predictions))
  print('Test RMSE: %.3f' % rmse)

arima("ABBV")