# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 09:58:04 2024

@author: kuldeep.rana
"""

import yaml
import json
import time
import warnings        
import math   
import sys
# from github import Github

import streamlit  as st
import numpy      as np
import pandas     as pd
import yfinance   as yf
import mplfinance as mpf
import pandas_ta  as ta
import matplotlib.pyplot as plt

import streamlit_authenticator as stauth
import plotly.graph_objects    as go

from plotly.subplots import make_subplots    
from scipy.signal import argrelextrema
from collections  import deque
from datetime     import datetime, timedelta
from yaml.loader  import SafeLoader
from streamlit_authenticator.utilities.exceptions import (CredentialsError,ForgotError,LoginError,RegisterError,ResetError,UpdateError) 

warnings.filterwarnings('ignore')

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Remove 'pre-authorized' from Authenticate constructor
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)




# authenticator.register_user(pre_authorized=['123@gmail.com'])
authenticator.login()


if st.session_state["authentication_status"]:
    authenticator.logout(location='sidebar')
    st.write(f'Welcome to AI ML *{st.session_state["name"]}*')      

    def get_date_range():
        today = datetime.today()
        start_dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(61)]  # Past 2 months
        end_dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(4)]  # Next 2 days
        return start_dates, end_dates
    
    # st.title("AI-ML Workflow")
    # st.sidebar.header("Instructions")
    # st.sidebar.markdown(""" Version 0.1.1
    # 1. **Select a ticker from the dropdown menu.**
    # 2. **Choose the desired time interval.**
    # 3. **Select the start and end dates for the data.**
    # 4. **Choose whether to analyze data using the default or multiple time frames(MTF).MTF means if you select time interval 5 min and MTF then data will be ploted for 5 min and calculations will be for 15 mins. Refer the below list for time frames ntervals = ['5m', '15m', '30m', '1h', '1d', '5d', '1wk', '1mo', '3mo','1m', '2m']**
    # 5. **Click the 'Plot' button to visualize the data.**
    # 6. **Note: For Forex (gold), use symbol 'GC=F'.**""")
    
    tickers = ['^NSEI','GC=F','^NSEBANK','ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJAJFINSV.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'BPCL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'INFY.NS', 'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'LTIM.NS', 'M&M.NS', 'MARUTI.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS', 'SHRIRAMFIN.NS', 'SUNPHARMA.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TCS.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS']
    ticker = st.selectbox("Select Ticker", tickers)
    intervals = ['5m', '15m', '30m', '1h', '1d', '5d', '1wk', '1mo', '3mo','1m', '2m']
    interval = st.selectbox("Select Interval", intervals)

    start_dates, end_dates = get_date_range()
    start_date = st.date_input("Start Date", datetime.today() - timedelta(days=3))
    end_date = st.date_input("End Date", datetime.today() + timedelta(days=1))

    timeframe_options = ['Default', 'MTF']  # Default and Multi-Timeframe (MTF)
    timeframe = st.selectbox("Timeframe Analysis", timeframe_options)

    def plot_data():
        
        yfObj = yf.Ticker(ticker)
        if timeframe == 'Default':
            data = yfObj.history(start=start_date, end=end_date, interval=interval)
        elif timeframe == 'MTF':
            current_interval_index = intervals.index(interval)
            next_interval_index = current_interval_index + 1 if current_interval_index < len(intervals) - 1 else current_interval_index
            next_interval = intervals[next_interval_index]
            data  = yfObj.history(start=start_date, end=end_date, interval=next_interval)
            data1 = yfObj.history(start=start_date, end=end_date, interval=interval)
        if data.empty:
            custom_markdown = """
            ## Error in Processing
            <font color='red'> **Data loading failed**,  contact site-owner *OR* check Network connection. </font>  <font size='6'>
            """
            st.markdown(custom_markdown, unsafe_allow_html=True)
            
            
        data.reset_index(inplace=True)

        if timeframe == 'Default' and interval in ["1d", "1wk", "1mo", "3mo"]:
            data.set_index('Date', inplace=True)
        elif timeframe == 'Default':
            data.set_index('Datetime', inplace=True)
        elif timeframe == 'MTF' and next_interval in ["1d", "1wk", "1mo", "3mo"]:
            data.set_index('Date', inplace=True)
        elif timeframe == 'MTF':
            data.set_index('Datetime', inplace=True)

        order = 4
        K     = 2

        data['local_max'] = data['Close'][(data['Close'].shift(1) < data['Close']) & (data['Close'].shift(-1) < data['Close'])]
        data['local_min'] = data['Close'][(data['Close'].shift(1) > data['Close']) & (data['Close'].shift(-1) > data['Close'])]
      
        high_idx = argrelextrema(data['Close'].values, np.greater, order=order)[0]
        highs = data.iloc[high_idx]['Close']
        extrema = []
        ex_deque = deque(maxlen=K)

        for i, idx in enumerate(high_idx):
            if i == 0:
                ex_deque.append(idx)
                continue
            if highs[i] < highs[i-1]:
                ex_deque.clear()

            ex_deque.append(idx)
            if len(ex_deque) == K:
                extrema.append(ex_deque.copy())

        close = data['Close'].values
        dates = data.index

        def getHigherLows(data: np.array, order=order, K=2):

            low_idx = argrelextrema(data, np.less, order=order)[0]
            lows = data[low_idx]
            extrema = []
            ex_deque = deque(maxlen=K)
            for i, idx in enumerate(low_idx):
                if i == 0:
                    ex_deque.append(idx)
                    continue
                if lows[i] < lows[i-1]:
                    ex_deque.clear()

                ex_deque.append(idx)

                if len(ex_deque) == K:
                    extrema.append(ex_deque.copy())

            return extrema

        def getLowerHighs(data: np.array, order=order, K=2):

            high_idx = argrelextrema(data, np.greater, order=order)[0]
            highs = data[high_idx]
            extrema = []
            ex_deque = deque(maxlen=K)
            for i, idx in enumerate(high_idx):
                if i == 0:
                    ex_deque.append(idx)
                    continue
                if highs[i] > highs[i-1]:
                    ex_deque.clear()

                ex_deque.append(idx)
                if len(ex_deque) == K:
                    extrema.append(ex_deque.copy())

            return extrema

        def getHigherHighs(data: np.array, order=order, K=2):

            high_idx = argrelextrema(data, np.greater, order=order)[0]
            highs = data[high_idx]
            extrema = []
            ex_deque = deque(maxlen=K)
            for i, idx in enumerate(high_idx):
                if i == 0:
                    ex_deque.append(idx)
                    continue
                if highs[i] < highs[i-1]:
                    ex_deque.clear()

                ex_deque.append(idx)
                if len(ex_deque) == K:
                    extrema.append(ex_deque.copy())

            return extrema

        def getLowerLows(data: np.array, order=order, K=2):

            low_idx = argrelextrema(data, np.less, order=order)[0]
            lows = data[low_idx]
            extrema = []
            ex_deque = deque(maxlen=K)
            for i, idx in enumerate(low_idx):
                if i == 0:
                    ex_deque.append(idx)
                    continue
                if lows[i] > lows[i-1]:
                    ex_deque.clear()

                ex_deque.append(idx)
                if len(ex_deque) == K:
                    extrema.append(ex_deque.copy())

            return extrema

        close = data['Close'].values
        dates = data.index

        hh = getHigherHighs(close, order, K)
        hl = getHigherLows(close, order, K)
        ll = getLowerLows(close, order, K)
        lh = getLowerHighs(close, order, K)

        def decode_peaks(df):    
            df_val = pd.DataFrame(df)
            df_val['value_0'] = np.nan
            df_val['value_1'] = np.nan
            for index, row in df_val.iterrows():
                index_0 = row[0]
                index_1 = row[1]        
                index_0 = int(index_0)
                index_1 = int(index_1)        
                if 0 <= index_0 < len(close) and 0 <= index_1 < len(close):
                    value_0 = close[index_0]
                    value_1 = close[index_1]            
                    df_val.at[index, 'value_0'] = value_0
                    df_val.at[index, 'value_1'] = value_1

            return df_val
        def calc_rangebreak(time_series:pd.Series):           
            timedeltas = time_series.diff()            
            if len(time_series) < 2:
                return []            
            missing_times = np.where([timedeltas > timedeltas.median()*1.5])[1]
            off = pd.Timedelta(seconds=0.0001)
            rb = [{'bounds': [str((time_series.iloc[t-1]+off)), str((time_series.iloc[t]-off))]} for t in missing_times]
            return rb
        def levels_cal(df,fibScale):
            df = df.copy()
            fib_levels = (0.24, 0.38, 0.82, 1.68, 2.3, 3.6, 4.0)
        
            fibLevel1 = fib_levels[0]
            fibLevel2 = fib_levels[1]
            fibLevel3 = fib_levels[2]
            fibLevel4 = fib_levels[3]
            fibLevel5 = fib_levels[4]
            fibLevel6 = fib_levels[5]
            fibLevel7 = fib_levels[6]
            today = pd.Timestamp.now().date()
            
            if today.weekday() in {5,6}:
               price_5_max =0
               price_5_min  = 0
            else:                                
                df.reset_index(inplace=True)
                df['datetime_column'] = pd.to_datetime(df['Datetime'])                    
                today_data = df[df['datetime_column'].dt.date == today]
                price_fut_high = today_data['high']
                price_fut_low  = today_data['low']                  
                price_5_max    = max(price_fut_high.head(12))
                price_5_min    = min(price_fut_low.head(12))
                price_difference = price_5_max - price_5_min
            
            range_3_Low    = price_5_min - ((price_5_max-price_5_min) * fibLevel3 * fibScale)
            range_6_Low    = price_5_min - ((price_5_max-price_5_min) * fibLevel2 * fibScale)
            range_3_High   = price_5_min + ((price_5_max-price_5_min) * fibLevel3 * fibScale)
            range_6_High   = price_5_min + ((price_5_max-price_5_min) * fibLevel2 * fibScale)
            range_8_Low    = price_5_min - ((price_5_max-price_5_min) * fibLevel4 * fibScale)
            range_9_Low    = price_5_min - ((price_5_max-price_5_min) * fibLevel5 * fibScale)
            range_8_High   = price_5_min + ((price_5_max-price_5_min) * fibLevel4 * fibScale)
            range_9_High   = price_5_min + ((price_5_max-price_5_min) * fibLevel5 * fibScale)
            range_7_Low    = price_5_min - ((price_5_max-price_5_min) * fibLevel4 * fibScale)
            range_10_Low   = price_5_min - ((price_5_max-price_5_min) * fibLevel5 * fibScale)
            range_7_High   = price_5_min + ((price_5_max-price_5_min) * fibLevel4 * fibScale)
            range_10_High  = price_5_min + ((price_5_max-price_5_min) * fibLevel5 * fibScale)
            range_1_Low    = price_5_min - ((price_5_max-price_5_min) * fibLevel1 * fibScale)
            range_1_high   = price_5_min + ((price_5_max-price_5_min) * fibLevel1 * fibScale)
            range_11_Low   = price_5_min - ((price_5_max-price_5_min) * fibLevel6 * fibScale)
            range_11_High  = price_5_min + ((price_5_max-price_5_min) * fibLevel6 * fibScale)
            range_12_Low   = price_5_min - ((price_5_max-price_5_min) * fibLevel7 * fibScale)
            range_12_High  = price_5_min + ((price_5_max-price_5_min) * fibLevel7 * fibScale)
            
            fib_df_main    = [range_3_Low,range_6_Low,range_3_High,range_6_High,range_8_Low,range_9_Low,range_8_High,range_9_High,
                              range_7_Low,range_10_Low,range_7_High,range_10_High,range_1_Low,range_1_high,range_11_Low,range_12_Low,range_11_High,range_12_High]
            
            return fib_df_main
           
        def __to_inc(x):
                incs = x[1:] - x[:-1]
                return incs

        def __to_pct(x):
                pcts = x[1:] / x[:-1] - 1.
                return pcts

        def __get_simplified_RS(series, kind):
                """
                Simplified version of rescaled range

                Parameters
                ----------

                series : array-like
                    (Time-)series
                kind : str
                    The kind of series (refer to compute_Hc docstring)
                """

                if kind == 'random_walk':
                    incs = __to_inc(series)
                    R = max(series) - min(series)  # range in absolute values
                    S = np.std(incs, ddof=1)
                elif kind == 'price':
                    pcts = __to_pct(series)
                    R = max(series) / min(series) - 1. # range in percent
                    S = np.std(pcts, ddof=1)
                elif kind == 'change':
                    incs = series
                    _series = np.hstack([[0.],np.cumsum(incs)])
                    R = max(_series) - min(_series)  # range in absolute values
                    S = np.std(incs, ddof=1)

                if R == 0 or S == 0:
                    return 0  # return 0 to skip this interval due the undefined R/S ratio

                return R / S

        def __get_RS(series, kind):
                """
                Get rescaled range (using the range of cumulative sum
                of deviations instead of the range of a series as in the simplified version
                of R/S) from a time-series of values.

                Parameters
                ----------

                series : array-like
                    (Time-)series
                kind : str
                    The kind of series (refer to compute_Hc docstring)
                """

                if kind == 'random_walk':
                    incs = __to_inc(series)
                    mean_inc = (series[-1] - series[0]) / len(incs)
                    deviations = incs - mean_inc
                    Z = np.cumsum(deviations)
                    R = max(Z) - min(Z)
                    S = np.std(incs, ddof=1)

                elif kind == 'price':
                    incs = __to_pct(series)
                    mean_inc = np.sum(incs) / len(incs)
                    deviations = incs - mean_inc
                    Z = np.cumsum(deviations)
                    R = max(Z) - min(Z)
                    S = np.std(incs, ddof=1)

                elif kind == 'change':
                    incs = series
                    mean_inc = np.sum(incs) / len(incs)
                    deviations = incs - mean_inc
                    Z = np.cumsum(deviations)
                    R = max(Z) - min(Z)
                    S = np.std(incs, ddof=1)

                if R == 0 or S == 0:
                    return 0  # return 0 to skip this interval due undefined R/S

                return R / S

        def compute_Hc(series, kind="random_walk", min_window=10, max_window=None, simplified=True):
                """
                Compute H (Hurst exponent) and C according to Hurst equation:
                E(R/S) = c * T^H

                Refer to:
                https://en.wikipedia.org/wiki/Hurst_exponent
                https://en.wikipedia.org/wiki/Rescaled_range
                https://en.wikipedia.org/wiki/Random_walk

                Parameters
                ----------

                series : array-like
                    (Time-)series

                kind : str
                    Kind of series
                    possible values are 'random_walk', 'change' and 'price':
                    - 'random_walk' means that a series is a random walk with random increments;
                    - 'price' means that a series is a random walk with random multipliers;
                    - 'change' means that a series consists of random increments
                        (thus produced random walk is a cumulative sum of increments);

                min_window : int, default 10
                    the minimal window size for R/S calculation

                max_window : int, default is the length of series minus 1
                    the maximal window size for R/S calculation

                simplified : bool, default True
                    whether to use the simplified or the original version of R/S calculation

                Returns tuple of
                    H, c and data
                    where H and c — parameters or Hurst equation
                    and data is a list of 2 lists: time intervals and R/S-values for correspoding time interval
                    for further plotting log(data[0]) on X and log(data[1]) on Y
                """

                if len(series)<10:
                    raise ValueError("Series length must be greater or equal to 100")

                ndarray_likes = [np.ndarray]
                if "pandas.core.series" in sys.modules.keys():
                    ndarray_likes.append(pd.core.series.Series)

                # convert series to numpy array if series is not numpy array or pandas Series
                if type(series) not in ndarray_likes:
                    series = np.array(series)

                if "pandas.core.series" in sys.modules.keys() and type(series) == pd.core.series.Series:
                    if series.isnull().values.any():
                        raise ValueError("Series contains NaNs")
                    series = series.values  # convert pandas Series to numpy array
                elif np.isnan(np.min(series)):
                    raise ValueError("Series contains NaNs")

                if simplified:
                    RS_func = __get_simplified_RS
                else:
                    RS_func = __get_RS


                err = np.geterr()
                np.seterr(all='raise')

                max_window = max_window or len(series)-1
                window_sizes = list(map(
                    lambda x: int(10**x),
                    np.arange(math.log10(min_window), math.log10(max_window), 0.25)))
                window_sizes.append(len(series))

                RS = []
                for w in window_sizes:
                    rs = []
                    for start in range(0, len(series), w):
                        if (start+w)>len(series):
                            break
                        _ = RS_func(series[start:start+w], kind)
                        if _ != 0:
                            rs.append(_)
                    RS.append(np.mean(rs))

                A = np.vstack([np.log10(window_sizes), np.ones(len(RS))]).T
                H, c = np.linalg.lstsq(A, np.log10(RS), rcond=-1)[0]
                np.seterr(**err)

                c = 10**c
                return H, c, [window_sizes, RS]

        def random_walk(length, proba=0.5, min_lookback=1, max_lookback=100, cumprod=False):
                """
                Generates a random walk series

                Parameters
                ----------

                proba : float, default 0.5
                    the probability that the next increment will follow the trend.
                    Set proba > 0.5 for the persistent random walk,
                    set proba < 0.5 for the antipersistent one

                min_lookback: int, default 1
                max_lookback: int, default 100
                    minimum and maximum window sizes to calculate trend direction
                cumprod : bool, default False
                    generate a random walk as a cumulative product instead of cumulative sum
                """

                assert(min_lookback>=1)
                assert(max_lookback>=min_lookback)

                if max_lookback > length:
                    max_lookback = length
                    warnings.warn("max_lookback parameter has been set to the length of the random walk series.")

                if not cumprod:  # ordinary increments
                    series = [0.] * length  # array of prices
                    for i in range(1, length):
                        if i < min_lookback + 1:
                            direction = np.sign(np.random.randn())
                        else:
                            lookback = np.random.randint(min_lookback, min(i-1, max_lookback)+1)
                            direction = np.sign(series[i-1] - series[i-1-lookback]) * np.sign(proba - np.random.uniform())
                        series[i] = series[i-1] + np.fabs(np.random.randn()) * direction
                else:  # percent changes
                    series = [1.] * length  # array of prices
                    for i in range(1, length):
                        if i < min_lookback + 1:
                            direction = np.sign(np.random.randn())
                        else:
                            lookback = np.random.randint(min_lookback, min(i-1, max_lookback)+1)
                            direction = np.sign(series[i-1] / series[i-1-lookback] - 1.) * np.sign(proba - np.random.uniform())
                        series[i] = series[i-1] * np.fabs(1 + np.random.randn()/1000. * direction)

                return series


        data.index = pd.to_datetime(data.index)
        data.columns = data.columns.str.lower()

        out_hh = [[(dates[i][0],close[i][0]),(dates[i][1],close[i][1])] for i in hh]
        out_hl = [[(dates[i][0],close[i][0]),(dates[i][1],close[i][1])] for i in hl]
        out_ll = [[(dates[i][0],close[i][0]),(dates[i][1],close[i][1])] for i in ll]
        out_lh = [[(dates[i][0],close[i][0]),(dates[i][1],close[i][1])] for i in lh]

        joined_output = out_hh + out_hl + out_ll + out_lh        
       
        def post_processing(data_list):        
            data            = [[x[1] for x in sublist] for sublist in data_list]
            timestamps      = [[x[0] for x in sublist] for sublist in data_list]
            data_flat       = [item for sublist in data for item in sublist]
            timestamps_flat = [item for sublist in timestamps for item in sublist]
            df = pd.DataFrame({'time': timestamps_flat, 'value': data_flat})
            return df
        def plot_plotly(data_plotly,asset_zone,macd,df_hh,df_hl,df_ll,df_lh,r_length,rsi_df):
            fig = make_subplots(rows=4, cols=1, shared_xaxes=True,shared_yaxes=True,
                  vertical_spacing=0.05,row_heights=[0.4, 0.1, 0.1, 0.1])
            fig.add_trace(go.Candlestick(
                    x=data_plotly.index,
                    open=data_plotly['open'],
                    high=data_plotly['high'],
                    low=data_plotly['low'],
                    close=data_plotly['close'],
                    name=f'{ticker}'), row=1, col=1 )            
            fig.add_trace(go.Scatter(
                    x=data_plotly.index,
                    y=[asset_zone[2],asset_zone[6]]*len(data_plotly), 
                    mode='markers',
                    line=dict(color="red",width=7, dash='dash'),
                    name='Zone1'), row=1, col=1)
            fig.add_trace(go.Scatter(
                    x=data_plotly.index,
                    y=[asset_zone[1],asset_zone[0]]*len(data_plotly), 
                    mode='markers',
                    line=dict(color="green",width=7, dash='dash'),
                    name='Zone1'), row=1, col=1)           
            if data_plotly['volume'].notna().any():
                fig.add_trace(go.Bar(
                    x=data_plotly.index,
                    y=data_plotly['volume'], 
                    marker_color=colors_vol,
                    name='Volume'), row=2, col=1) 
                fig.add_trace(go.Scatter(
                    x=data_plotly.index,
                    y=rsi_df.values, 
                    mode='lines',
                    line=dict(color="yellow",width=3),
                    name='Vol_ma'), row=2, col=1)            
            fig.add_trace(go.Bar(
                    x=data_plotly.index,
                    y=macd[f'MACD_{fast}_{slow}_{signal}'], 
                    marker_color=colors_macd,
                    name='MACD_bar'), row=3, col=1)
            fig.add_trace(go.Scatter(
                    x=data_plotly.index,
                    y=macd[f'MACDs_{fast}_{slow}_{signal}'], 
                    mode='lines',
                    line=dict(color='yellow',width=3),
                    name='MACD_signal'), row=3, col=1)
            fig.add_trace(go.Scatter(
                    x=data_plotly.index,
                    y=macd[f'MACDh_{fast}_{slow}_{signal}'], 
                    mode='lines',
                    line=dict(color='blue',width=3),
                    name='MACD_h'), row=3, col=1)
            fig.add_trace(go.Scatter(
                    x=data_plotly.index,
                    y=rsi.values, 
                    mode='lines',
                    line=dict(color="teal"),
                    name='RSI'), row=4, col=1)
            fig.add_trace(go.Scatter(
                    x=data_plotly.index,
                    y=[80] * len(data_plotly), 
                    mode='lines',
                    line=dict(color="red",width=3, dash='dash'),
                    name='RSI_High'), row=4, col=1)
            fig.add_trace(go.Scatter(
                    x=data_plotly.index,
                    y=[20] * len(data_plotly), 
                    mode='lines',
                    line=dict(color="green",width=3, dash='dash'),
                    name='RSI_Low'), row=4, col=1)
            fig.add_trace(go.Scatter(
                    x=data_plotly.index,
                    y=[50] * len(data_plotly), 
                    mode='lines',
                    line=dict(color="blue",width=2, dash='dash'),
                    name='RSI_mid'), row=4, col=1)           
            fig.add_trace(go.Scatter(
                    x=df_hh['time'],
                    y=df_hh['value'],
                    mode='markers+lines',
                    marker=dict(color='red', size=8),
                    name='Higher High'),row=1, col=1)
            fig.add_trace(go.Scatter(
                    x=df_lh['time'],
                    y=df_lh['value'],
                    mode='markers+lines',
                    marker=dict(color='orange', size=8),
                    name='Lower High'),row=1, col=1)
            fig.add_trace(go.Scatter(
                    x=df_ll['time'],
                    y=df_ll['value'],
                    mode='markers+lines',
                    marker=dict(color='green', size=8),
                    name='Lower Low'),row=1, col=1)
            fig.add_trace(go.Scatter(
                    x=df_hl['time'],
                    y=df_hl['value'],
                    mode='markers+lines',
                    marker=dict(color='blue', size=8),
                    name='Higher Low'),row=1, col=1)
            
            # fig.add_trace(go.Scatter(
            #     x=data_plotly.index,
            #     y=df_stats['median'],
            #     mode='lines',
            #     name='Median'), row=5, col=1)            
            # fig.add_trace(go.Scatter(
            #     x=data_plotly.index,
            #     y=df_stats['quantile'],
            #     mode='lines',
            #     name='Quantile'), row=6, col=1)            
            # fig.add_trace(go.Scatter(
            #     x=data_plotly.index,
            #     y=df_stats['TOS_STDEVALL_20_LR'],
            #     mode='lines',
            #     name='TOS_STDEVALL_20_LR'), row=7, col=1)            
            # fig.add_trace(go.Scatter(
            #     x=data_plotly.index,
            #     y=df_stats['TOS_STDEVALL_20_L_1'],
            #     mode='lines',
            #     name='TOS_STDEVALL_20_L_1'), row=8, col=1)            
            # fig.add_trace(go.Scatter(
            #     x=data_plotly.index,
            #     y=df_stats['TOS_STDEVALL_20_U_1'],
            #     mode='lines',
            #     name='TOS_STDEVALL_20_U_1'), row=9, col=1)            
            # fig.add_trace(go.Scatter(
            #     x=data_plotly.index,
            #     y=df_stats['TOS_STDEVALL_20_L_2'],
            #     mode='lines',
            #     name='TOS_STDEVALL_20_L_2'), row=10, col=1)            
            # fig.add_trace(go.Scatter(
            #     x=data_plotly.index,
            #     y=df_stats['TOS_STDEVALL_20_U_2'],
            #     mode='lines',
            #     name='TOS_STDEVALL_20_U_2'), row=11, col=1)            
            # fig.add_trace(go.Scatter(
            #     x=data_plotly.index,
            #     y=df_stats['TOS_STDEVALL_20_L_3'],
            #     mode='lines',
            #     name='TOS_STDEVALL_20_L_3'), row=12, col=1)            
            # fig.add_trace(go.Scatter(
            #     x=data_plotly.index,
            #     y=df_stats['TOS_STDEVALL_20_U_3'],
            #     mode='lines',
            #     name='TOS_STDEVALL_20_U_3'), row=13, col=1)
            
            fig.update_layout(
                    title='Asset Data',
                    xaxis_title='time',
                    yaxis_title='values',
                    template='seaborn' ,
                    xaxis_rangeslider_visible=False,
                    width=1200,height=1600)
            (fig.update_xaxes(rangebreaks=rb_bounds))
            fig.update_yaxes(title_text='Volume', row=2, col=1) 
            fig.update_yaxes(title_text=f'MACD_{fast}_{slow}_{signal}', row=3, col=1)  
            fig.update_yaxes(title_text=f'RSI_{r_length}',    row=4, col=1)
            fig.update_yaxes(title_text=f'RSI_{r_length}',    row=5, col=1)

            
            st.plotly_chart(fig)
        st.set_option('deprecation.showPyplotGlobalUse', False)                     

        df_hh   = post_processing(out_hh)
        df_ll   = post_processing(out_ll)
        df_lh   = post_processing(out_hl)
        df_hl   = post_processing(out_lh)
        
        out_hh_ = [dates[i][1] for i in hh]
        out_hl_ = [dates[i][1] for i in hl]
        out_ll_ = [dates[i][1] for i in ll]
        out_lh_ = [dates[i][1] for i in lh]
        
        df    = data.copy()
        label = []
        for i in range(len(df)):
            if df.index[i] in out_hh_:
                label.append('HH')
            elif df.index[i] in out_hl_:
                label.append('HL')
            elif df.index[i] in out_ll_:
                label.append('LL')    
            elif df.index[i] in out_lh_:
                label.append('LH')    
            else:
                label.append(None)
                
        df['label'] = label
        fast = 12
        slow = 26
        signal = 9
        r_length = 13
        
        if timeframe == 'Default':                        
            data_plotly = data.copy() 

            data_plotly1 = data.copy()           
            close_prices = data_plotly1['close'].values            
            H, c, data_ = compute_Hc(close_prices,min_window=10)            
            st.title(f'Trend Strength :-{H,c}')
# =============================================================================
#             fig, ax = plt.subplots()
#             ax.plot(data_[0], c * data_[0] ** H, color="purple")
#             ax.set_xlabel('Time interval')
#             ax.set_ylabel('R/S ratio')
#             ax.set_title(f'Hurst Exponent: {H:.4f}')
#             
#             # Display the plot in Streamlit
#             st.pyplot(fig)
# =============================================================================
            

            
            macd        = data_plotly.ta.macd(close='close', fast=fast, slow=slow, signal=signal, append=True)
            rsi         = data_plotly.ta.rsi(close='close', length=r_length) 
            vol_mvg     = ta.ema(data_plotly['volume'],length=20)
            
            df_stats    = data.copy()
  
            df_stats['median'] = ta.median(df_stats['close'], length=20)
            df_stats['quantile'] = ta.quantile(df_stats['close'], q=0.25, length=20)
            tos_stdevall_df = ta.tos_stdevall(df_stats['close'], length=20)
            tos_columns=['TOS_STDEVALL_20_LR','TOS_STDEVALL_20_L_1','TOS_STDEVALL_20_U_1','TOS_STDEVALL_20_L_2','TOS_STDEVALL_20_U_2','TOS_STDEVALL_20_L_3','TOS_STDEVALL_20_U_3']
            df_stats[tos_columns] = tos_stdevall_df[tos_columns]
            
            rb_bounds   = calc_rangebreak(data_plotly.index.to_series())
            fibScale    = 2.1
            asset_zone  = levels_cal(data_plotly,fibScale)       
            
            colors_macd = ['red' if value < 0 else 'green' for value in data_plotly[f'MACD_{fast}_{slow}_{signal}']]
            colors_vol  = ['teal' if close > prev_close else 'red'
                            for close, prev_close in zip(data_plotly['close'], data_plotly['close'].shift(1, fill_value=data_plotly['close'].iloc[0]))]
            
            plot_plotly(data_plotly,asset_zone,macd,df_hh,df_hl,df_ll,df_lh,r_length,vol_mvg)
            # fig = mpf.plot(data, alines=joined_output, type='candle', style='starsandstripes')#, volume=False,figsize=(20, 12),tight_layout=True)
       
        elif timeframe == 'MTF':            
            data_plotly         = data1.copy()
            
            data_plotly.columns = [col.lower() for col in data_plotly.columns]
            macd        = data_plotly.ta.macd(close='close', fast=fast, slow=slow, signal=signal, append=True)
            rsi         = data_plotly.ta.rsi(close='close', length=r_length) 
            vol_mvg     = ta.ema(data_plotly['volume'],length=20)
            
            df_stats    = data.copy()  
            df_stats['median'] = ta.median(df_stats['close'], length=20)
            df_stats['quantile'] = ta.quantile(df_stats['close'], q=0.25, length=20)
            tos_stdevall_df = ta.tos_stdevall(df_stats['close'], length=20)
            tos_columns=['TOS_STDEVALL_20_LR','TOS_STDEVALL_20_L_1','TOS_STDEVALL_20_U_1','TOS_STDEVALL_20_L_2','TOS_STDEVALL_20_U_2','TOS_STDEVALL_20_L_3','TOS_STDEVALL_20_U_3']
            df_stats[tos_columns] = tos_stdevall_df[tos_columns]            
            rb_bounds   = calc_rangebreak(data_plotly.index.to_series())
            fibScale    = 2.6
            asset_zone  = levels_cal(data_plotly,fibScale)            
            colors_macd = ['red' if value < 0 else 'green' for value in data_plotly[f'MACD_{fast}_{slow}_{signal}']]
            colors_vol  = ['teal' if close > prev_close else 'red'
                            for close, prev_close in zip(data_plotly['close'], data_plotly['close'].shift(1, fill_value=data_plotly['close'].iloc[0]))]
                                 
            plot_plotly(data_plotly,asset_zone,macd,df_hh,df_hl,df_ll,df_lh,r_length,vol_mvg)
            
            fig = mpf.plot(data1, alines=joined_output, type='candle', style='starsandstripes', volume=False,)# figsize=(11, 22),tight_layout=True)     
            
        # st.pyplot(fig)
          
    if st.button("Plot"):
            plot_data()
        
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')

# """ Below code is for security: refer :https://github.com/mkhorasani/Streamlit-Authenticator/tree/main?tab=readme-ov-file#authenticateregister_user"""
st.sidebar.header("User Access Actions")

# g = Github("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")

# Access the repository and file
# repo = g.get_repo("kulrm23/stream_lit_deploy")
# file = repo.get_contents("config.yaml")

# Load the existing YAML content
# existing_content = yaml.safe_load(file.decoded_content)

show_register_form = st.sidebar.checkbox("New Registration")
if show_register_form:
    try:         
        (email_of_registered_user,
         username_of_registered_user,
         name_of_registered_user) = authenticator.register_user(pre_authorization=False, location='sidebar')

        if email_of_registered_user:
            st.success('User registered successfully')

            # Append the new user credentials to the existing 'usernames' section
            # if 'usernames' not in existing_content['credentials']:
            #     existing_content['credentials']['usernames'] = {}
            # existing_content['credentials']['usernames'][username_of_registered_user] = {
            #     'email': email_of_registered_user,
            #     'failed_login_attempts': 0,
            #     'logged_in': False,
            #     'name': name_of_registered_user,
            #     'password': ''  # You can set the password here if applicable
            # }

            # # Dump the updated YAML content to a string
            # updated_content = yaml.dump(existing_content, default_flow_style=False)

            # Commit and push the changes back to the GitHub repository
            # repo.update_file("config.yaml", "Updated user credentials", updated_content, file.sha)
            
    except RegisterError as e:
        st.error(e)
           
show_forgotten_password_form = st.sidebar.checkbox("Forgot Password")
if show_forgotten_password_form:
    try:
        (username_of_forgotten_password,
            email_of_forgotten_password,
            new_random_password) = authenticator.forgot_password(location='sidebar')
        if username_of_forgotten_password:
            st.success('New password sent securely')
            with open('config.yaml', 'w', encoding='utf-8') as file:
                yaml.dump(config, file, default_flow_style=False)  
            
        elif not username_of_forgotten_password:
            st.error('Username not found')
    except ForgotError as e:
        st.error(e)
        
        
show_username_password_form = st.sidebar.checkbox("Forgot User-name")
if show_username_password_form:    
    try:
        (username_of_forgotten_username,
            email_of_forgotten_username) = authenticator.forgot_username(location='sidebar')
        if username_of_forgotten_username:
            st.success('Username sent securely')
            with open('config.yaml', 'w', encoding='utf-8') as file:
                yaml.dump(config, file, default_flow_style=False)  
            # Username to be transferred to the user securely
        elif not username_of_forgotten_username:
            st.error('Email not found')
    except ForgotError as e:
        st.error(e)

        
show_update_user_form = st.sidebar.checkbox("Change User-name")
if show_update_user_form:
    if st.session_state["authentication_status"]:
        try:
            if authenticator.update_user_details(st.session_state["username"]):
                st.success('Entries updated successfully')
                with open('config.yaml', 'w', encoding='utf-8') as file:
                    yaml.dump(config, file, default_flow_style=False)  
        except UpdateError as e:
            st.error(e)



show_Reset_password_form = st.sidebar.checkbox("Reset Password")
if show_Reset_password_form:
    if st.session_state["authentication_status"]:
        try:
            if authenticator.reset_password(st.session_state["username"]):
                st.success('Password modified successfully')
                with open('config.yaml', 'w', encoding='utf-8') as file:
                    yaml.dump(config, file, default_flow_style=False)  
        except ResetError as e:
            st.error(e)
        except CredentialsError as e:
            st.error(e)
       

# with open('config.yaml', 'w', encoding='utf-8') as file:
#     yaml.dump(config, file, default_flow_style=False)   
    
# df_stats['skew'] = ta.skew(df_stats['close'], length=20)
# df_stats['stdev'] = ta.stdev(df_stats['close'], length=20)
# df_stats['variance'] = ta.variance(df_stats['close'], length=20)
# df_stats['zscore'] = ta.zscore(df_stats['close'], length=20)
# df_stats['entropy'] = ta.entropy(df_stats['close'], length=20)
# df_stats['kurtosis'] = ta.kurtosis(df_stats['close'], length=20)
# df_stats['mad'] = ta.mad(df_stats['close'], length=20)
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Assuming df_stats is your DataFrame
# # Calculate correlation matrix
# correlation_matrix = df_stats.corr()

# # Extract correlation of 'close' with all other columns
# correlation_with_close = correlation_matrix['close']

# # Plot heatmap of correlation matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
# plt.title('Correlation Heatmap')
# st.pyplot()

# # Plot heatmap of correlation with 'close'
# plt.figure(figsize=(8, 6))
# sns.heatmap(correlation_with_close.to_frame().T, annot=True, cmap='coolwarm', linewidths=0.5)
# plt.title('Correlation of "close" with Other Columns')
# st.pyplot()

# =============================================================================
# """ Below code is using light-weight charts
# 
# data['time'] = data.index
# data_lt = data.copy()
# data_lt.reset_index(inplace=True,drop=True)
# 
# columns_to_drop = ['dividends', 'stock splits',]
# df = data_lt.drop(columns=columns_to_drop)
#            
# from streamlit_lightweight_charts import renderLightweightCharts
# import pandas_ta as ta
# # import streamlit_lightweight_charts.dataSamples as data
#    
# COLOR_BULL = 'rgba(38,166,154,0.9)' # #26a69a
# COLOR_BEAR = 'rgba(239,83,80,0.9)'  # #ef5350
# 
# # joined_output1    
# # joined_output1['time'] = joined_output1['time'].view('int64') // 10**9 
# # joined_output1   = json.loads(joined_output1.filter(['time','value'], axis=1).to_json(orient = "records") )
# 
# df['time'] = df['time'].view('int64') // 10**9  # We will use time in UNIX timestamp
# df['color'] = np.where(  df['open'] > df['close'], COLOR_BEAR, COLOR_BULL)  # bull or bear
# df.ta.macd(close='close', fast=6, slow=12, signal=5, append=True) 
# 
# macd_fast = json.loads(df.rename(columns={"MACDh_6_12_5": "value"}).to_json(orient = "records"))
# macd_slow = json.loads(df.rename(columns={"MACDs_6_12_5": "value"}).to_json(orient = "records"))
# df['color'] = np.where(  df['MACD_6_12_5'] > 0, COLOR_BULL, COLOR_BEAR) 
# macd_hist   = json.loads(df.rename(columns={"MACD_6_12_5": "value"}).to_json(orient = "records"))
# 
# candles   = json.loads(df.filter(['time','open','high','low','close'], axis=1).to_json(orient = "records") )
# volume    = json.loads(df.filter(['time','volume'], axis=1).rename(columns={"volume": "value",}).to_json(orient = "records") )
# macd_fast = json.loads(df.filter(['time','macd_fast'], axis=1).rename(columns={"macd_fast": "value"}).to_json(orient = "records"))
# macd_slow = json.loads(df.filter(['time','macd_slow'], axis=1).rename(columns={"macd_slow": "value"}).to_json(orient = "records"))
# 
#                         
# # df_hh['time'] = df_hh['time'].view('int64') // 10**9 
# # df_hh   = json.loads(df_hh.filter(['time','value'], axis=1).to_json(orient = "records") ) 
# 
# # df_ll['time'] = df_ll['time'].view('int64') // 10**9 
# # df_ll   = json.loads(df_ll.filter(['time','value'], axis=1).to_json(orient = "records") ) 
# 
# # df_lh['time'] = df_lh['time'].view('int64') // 10**9 
# # df_lh   = json.loads(df_lh.filter(['time','value'], axis=1).to_json(orient = "records") ) 
# 
# # df_hl['time'] = df_hl['time'].view('int64') // 10**9 
# # df_hl   = json.loads(df_hl.filter(['time','value'], axis=1).to_json(orient = "records") ) 
# 
# # overlaidAreaSeriesOptions = {
# #     "height": 400,
# #     "rightPriceScale": {
# #         "scaleMargins": {
# #             "top":"",
# #             "bottom": "",
# #         },
# #         "mode": 0, # PriceScaleMode: 0-Normal, 1-Logarithmic, 2-Percentage, 3-IndexedTo100
# #         "borderColor": 'rgba(197, 203, 206, 0.4)',
# #     },
# #     "timeScale": {
# #         "borderColor": 'rgba(197, 203, 206, 0.4)',
# #     },
# #     "layout": {
# #         "background": {
# #             "type": 'solid',
# #             "color": '#100841'
# #         },
# #         "textColor": '#ffffff',
# #     },
# #     "grid": {
# #         "vertLines": {
# #             "color": 'rgba(197, 203, 206, 0.4)',
# #             "style": 1, # LineStyle: 0-Solid, 1-Dotted, 2-Dashed, 3-LargeDashed
# #         },
# #         "horzLines": {
# #             "color": 'rgba(197, 203, 206, 0.4)',
# #             "style": 1, # LineStyle: 0-Solid, 1-Dotted, 2-Dashed, 3-LargeDashed
# #         }
# #     }
# # }
# 
# # seriesOverlaidChart = [
# #     {
# #         "type": 'Line',
# #         "data": joined_output1,
# #         "options": {
# #             "topColor": 'rgba(255, 192, 0, 0.7)',
# #             "bottomColor": 'rgba(255, 192, 0, 0.3)',
# #             "lineColor": 'rgba(255, 192, 0, 1)',
# #             "lineWidth": 2,
# #         }
# #     },
#       
#                     
#     # {
#     #     "type": 'Line',
#     #     "data": df_lh,
#     #     "options": {
#     #         "topColor": 'rgba(67, 83, 254, 0.7)',
#     #         "bottomColor": 'rgba(67, 83, 254, 0.3)',
#     #         "lineColor": 'rgba(67, 83, 254, 1)',
#     #         "lineWidth": 2,
#     #     }
#     # },
#     # {
#     #     "type": 'Line',
#     #     "data": df_ll,
#     #     "options": {
#     #         "topColor": 'rgba(67, 83, 254, 0.7)',
#     #         "bottomColor": 'rgba(67, 83, 254, 0.3)',
#     #         "lineColor": 'rgba(67, 83, 254, 1)',
#     #         "lineWidth": 2,
#     #     }
#     # },
#     # {
#     #     "type": 'Line',
#     #     "data": df_hl,
#     #     "options": {
#     #         "topColor": 'rgba(67, 83, 254, 0.7)',
#     #         "bottomColor": 'rgba(67, 83, 254, 0.3)',
#     #         "lineColor": 'rgba(67, 83, 254, 1)',
#     #         "lineWidth": 2,
#     #     }
#     # }
# # ]
#         
# # st.subheader("Overlaid Series with Markers")
# 
# # renderLightweightCharts([
# #     {
# #         "chart": overlaidAreaSeriesOptions,
# #         "series": seriesOverlaidChart
# #     }
# # ], 'overlaid')
# 
# 
# 
# # chartMultipaneOptions = [
# #     {
# #         "width": 1200,
# #         "height": 800,
# #         "layout": {
# #             "background": {
# #                 "type": "solid",
# #                 "color": "Black"
# #             },
# #             "textColor": "white"
# #         },
# #         "grid": {
# #             "vertLines": {
# #                 "color": "rgba(197, 203, 206, 0.5)"
# #             },
# #             "horzLines": {
# #                 "color": "rgba(197, 203, 206, 0.5)"
# #             }
# #         },
# #         "crosshair": {
# #             "mode": 0
# #         },
# #         "priceScale": {
# #             "borderColor": "rgba(197, 203, 206, 0.8)"
# #         },
# #         "timeScale": {
# #             "borderColor": "rgba(197, 203, 206, 0.8)",
# #             "barSpacing": 10,
# #             "minBarSpacing": 8,
# #             "timeVisible": True,
# #             "secondsVisible": False
# #         },
# #         "watermark": {
# #             "visible": True,
# #             "fontSize": 48,
# #             "horzAlign": "center",
# #             "vertAlign": "center",
# #             "color": "rgba(171, 71, 188, 0.3)",
# #             "text": "data"
# #         }
# #     },
# #     {
# #         "width": 1200,
# #         "height": 400,
# #         "layout": {
# #             "background": {
# #                 "type": "solid",
# #                 "color": "transparent"
# #             },
# #             "textColor": "black"
# #         },
# #         "grid": {
# #             "vertLines": {
# #                 "color": "rgba(42, 46, 57, 0)"
# #             },
# #             "horzLines": {
# #                 "color": "rgba(42, 46, 57, 0.6)"
# #             }
# #         },
# #         "timeScale": {
# #             "visible": False
# #         },
# #         "watermark": {
# #             "visible": True,
# #             "fontSize": 18,
# #             "horzAlign": "left",
# #             "vertAlign": "top",
# #             "color": "rgba(171, 71, 188, 0.7)",
# #             "text": "Volume"
# #         }
# #     },
# #     {
# #         "width": 1200,
# #         "height": 400,
# #         "layout": {
# #             "background": {
# #                 "type": "transparent",
# #                 "color": "black"
# #             },
# #             "textColor": "black"
# #         },
# #         "timeScale": {
# #             "visible": False
# #         },
# #         "watermark": {
# #             "visible": True,
# #             "fontSize": 18,
# #             "horzAlign": "left",
# #             "vertAlign": "center",
# #             "color": "rgba(171, 71, 188, 0.7)",
# #             "text": "MACD"
# #         }
# #     }
# # ]
# 
# 
# # seriesCandlestickChart = [
# #     {
# #         "type": 'Candlestick',
# #         "data": candles,
# #         "options": {
# #             "upColor": COLOR_BULL,
# #             "downColor": COLOR_BEAR,
# #             "borderVisible": False,
# #             "wickUpColor": COLOR_BULL,
# #             "wickDownColor": COLOR_BEAR
# #         }
# #     }
# # ]
# 
# # seriesVolumeChart = [
# #     {
# #         "type": 'Histogram',
# #         "data": volume,
# #         "options": {
# #             "priceFormat": {
# #                 "type": 'volume',
# #             },
# #             "priceScaleId": "" # set as an overlay setting,
# #         },
# #         "priceScale": {
# #             "scaleMargins": {
# #                 "top": 0,
# #                 "bottom": 0,
# #             },
# #             "alignLabels": False
# #         }
# #     }
# # ]
# 
# # seriesMACDchart = [
# #     {
# #         "type": 'Line',
# #         "data": macd_fast,
# #         "options": {
# #             "color": 'blue',
# #             "lineWidth": 2
# #         }
# #     },
# #     {
# #         "type": 'Line',
# #         "data": macd_slow,
# #         "options": {
# #             "color": 'green',
# #             "lineWidth": 2
# #         }
# #     },
# #     {
# #         "type": 'Histogram',
# #         "data": macd_hist,
# #         "options": {
# #             # "color": 'red',
# #             "lineWidth": 1
# #         }
# #     }
# # ]
# 
# 
# 
# # st.subheader("Multipane Chart ")
# # # joined_output1
# # renderLightweightCharts([
# #     {
# #         "chart": chartMultipaneOptions[0],
# #         "series": seriesCandlestickChart
# #     },
# #     {
# #         "chart": chartMultipaneOptions[1],
# #         "series": seriesVolumeChart
# #     },
# #     {
# #         "chart": chartMultipaneOptions[2],
# #         "series": seriesMACDchart
# #     },
#   
# 
# # ], 'area')
# """
# =============================================================================
