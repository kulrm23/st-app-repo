# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:33:34 2024
@author: kuldeep.rana
"""

import time
import warnings
import pandas as pd
import yfinance as yf
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
from collections import deque
from pandas.tseries.offsets import BusinessDay
import pytz

warnings.filterwarnings('ignore')

# Set page configuration as the first Streamlit command
st.set_page_config(page_title="Financial Analysis Platform", layout="centered")

# Custom CSS for mobile responsiveness
st.markdown("""
    <style>
    /* Ensure main content is responsive */
    .main .block-container {
        padding: 1rem;
        max-width: 100%;
    }
    /* Adjust font sizes for mobile */
    @media (max-width: 768px) {
        .stMarkdown, .stText, .stButton > button, .stSelectbox, .stDateInput {
            font-size: 14px !important;
        }
        /* Make dataframe scrollable on mobile */
        .stDataFrame {
            overflow-x: auto;
            width: 100%;
        }
        /* Reduce padding for inputs */
        .stSelectbox, .stDateInput, .stCheckbox {
            padding: 0.5rem 0;
        }
        /* Adjust calculation summary text */
        .calc-summary {
            font-size: 14px !important;
            line-height: 1.2;
        }
    }
    /* Improve button and input spacing */
    .stButton > button {
        width: 100%;
        margin-top: 0.5rem;
    }
    /* Ensure Plotly chart is responsive */
    .js-plotly-plot .plotly {
        width: 100% !important;
        height: auto !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_data(ticker, interval, start, end):
    try:
        yfObj = yf.Ticker(ticker)
        # Ensure end date is at least one minute after start for intraday data
        if start == end:
            end = (pd.to_datetime(start) + timedelta(days=1)).strftime("%Y-%m-%d")
        # Restrict to 7 days for 5m interval
        if interval == '5m' and (pd.to_datetime(end) - pd.to_datetime(start)).days > 7:
            start = (pd.to_datetime(end) - timedelta(days=7)).strftime("%Y-%m-%d")
        data = yfObj.history(start=start, end=end, interval=interval)
        if data.empty or len(data) < 2:  # Ensure data has enough points
            st.warning(f"No valid data retrieved for {ticker} with interval {interval}.")
            return None
        return data
    except Exception as e:
        st.warning(f"Error retrieving data for {ticker}: {str(e)}")
        return None

def get_date_range():
    today = datetime.today()
    start_dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]  # Past 7 days for 5m
    end_dates = [today.strftime("%Y-%m-%d")]  # Only today
    return start_dates, end_dates

def calc_rangebreak(time_series: pd.Series):
    timedeltas = time_series.diff()
    if len(time_series) < 2:
        return []
    missing_times = np.where([timedeltas > timedeltas.median() * 1.5])[1]
    off = pd.Timedelta(seconds=0.0001)
    rb = [{'bounds': [str((time_series.iloc[t-1] + off)), str((time_series.iloc[t] - off))]} for t in missing_times]
    return rb

def run_analysis(ticker, ticker_index, interval, start_date, end_date):
    # Dictionary to store dataframes with ticker names
    data_dict = {}
    failed_tickers = []

    # Fetch data for tickers
    for t in ticker:
        data = get_data(t, interval, start_date, end_date)
        if data is not None:
            data_dict[t] = data
        else:
            failed_tickers.append(t)

    # Fetch data for ticker_index
    for t in ticker_index:
        data = get_data(t, interval, start_date, end_date)
        if data is not None:
            data_dict[t] = data
        else:
            failed_tickers.append(t)

    if not data_dict:
        st.error("No valid data retrieved for any tickers. Please check ticker symbols, date range, or interval.")
        return

    if failed_tickers:
        st.warning(f"Failed to retrieve data for: {', '.join(failed_tickers)}")

    def cal_data(data, ticker_name):
        try:
            order = 4
            K = 2

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

            data.index = pd.to_datetime(data.index)
            data.columns = data.columns.str.lower()

            out_hh = [[(dates[i][0], close[i][0]), (dates[i][1], close[i][1])] for i in hh]
            out_hl = [[(dates[i][0], close[i][0]), (dates[i][1], close[i][1])] for i in hl]
            out_ll = [[(dates[i][0], close[i][0]), (dates[i][1], close[i][1])] for i in ll]
            out_lh = [[(dates[i][0], close[i][0]), (dates[i][1], close[i][1])] for i in lh]

            joined_output = out_hh + out_hl + out_ll + out_lh

            def post_processing(data_list):
                data = [[x[1] for x in sublist] for sublist in data_list]
                timestamps = [[x[0] for x in sublist] for sublist in data_list]
                data_flat = [item for sublist in data for item in sublist]
                timestamps_flat = [item for sublist in timestamps for item in sublist]
                df = pd.DataFrame({'time': timestamps_flat, 'value': data_flat})
                return df

            df_hh = post_processing(out_hh)
            df_ll = post_processing(out_ll)
            df_lh = post_processing(out_hl)
            df_hl = post_processing(out_lh)

            out_hh_ = [dates[i][1] for i in hh]
            out_hl_ = [dates[i][1] for i in hl]
            out_ll_ = [dates[i][1] for i in ll]
            out_lh_ = [dates[i][1] for i in lh]

            df = data.copy()
            label = []
            for i in range(len(df)):
                if df.index[i] in out_hl_:
                    label.append('HL')
                elif df.index[i] in out_lh_:
                    label.append('LH')
                else:
                    label.append(None)

            df['label'] = label
            return df
        except Exception as e:
            st.warning(f"Error processing data for {ticker_name}: {str(e)}")
            return None

    # Process data for each ticker
    maxima_dict = {}
    for t, data in data_dict.items():
        maxima = cal_data(data, t)
        if maxima is not None and not maxima.empty:
            maxima_dict[t] = maxima
        else:
            failed_tickers.append(t)

    if not maxima_dict:
        st.error("No valid processed data available. Analysis cannot proceed.")
        return

    # Define dataframes lists
    dataframes = [maxima_dict.get(t) for t in ticker if t in maxima_dict]
    dataframes = [df for df in dataframes if df is not None]  # Filter out None
    dataframes_weighted = [maxima_dict.get(t) for t in [
        '^NSEI', '^NSEBANK', 'AXISBANK.NS', 'BHARTIARTL.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
        'INFY.NS', 'ITC.NS', 'RELIANCE.NS', 'SBIN.NS', 'TCS.NS', 'LT.NS'
    ] if t in maxima_dict]
    dataframes_weighted = [df for df in dataframes_weighted if df is not None]
    dataframes_index = [maxima_dict.get(t) for t in ticker_index if t in maxima_dict]
    dataframes_index = [df for df in dataframes_index if df is not None]
    dataframes_index_weighted = [maxima_dict.get(t) for t in [
        '^CNXAUTO', '^CNXFMCG', '^CNXENERGY', '^CNXIT'
    ] if t in maxima_dict]
    dataframes_index_weighted = [df for df in dataframes_index_weighted if df is not None]

    def rename_and_extract_label(df, name):
        try:
            df = df.rename(columns={'label': f'{name}_label'})
            return df[[f'{name}_label']]
        except Exception as e:
            st.warning(f"Error renaming label for {name}: {str(e)}")
            return None

    def calculate_active_occurrences(dataframes, dataframe_names):
        renamed_dfs = []
        for df, name in zip(dataframes, dataframe_names):
            if df is not None and not df.empty:
                renamed_df = rename_and_extract_label(df, name)
                if renamed_df is not None and not renamed_df.empty:
                    renamed_dfs.append(renamed_df)

        if not renamed_dfs:
            st.error("No valid dataframes to concatenate. Analysis cannot proceed.")
            return None

        try:
            merged_df = pd.concat(renamed_dfs, axis=1)
            merged_df['Long'] = 0
            merged_df['Short'] = 0

            for index, row in merged_df.iterrows():
                lh_count = 0
                hl_count = 0
                for col in merged_df.columns:
                    if '_label' in col:
                        if row[col] == 'LH':
                            lh_count += 1
                        elif row[col] == 'HL':
                            hl_count += 1
                        merged_df.at[index, 'Short'] = lh_count
                        merged_df.at[index, 'Long'] = hl_count

            return merged_df
        except Exception as e:
            st.error(f"Error concatenating dataframes: {str(e)}")
            return None

    # Define dataframe names (using ticker symbols without special characters)
    dataframe_names = [t.replace('^', '').replace('.', '_') for t in ticker if t in maxima_dict]
    dataframe_names_index = [t.replace('^', '').replace('.', '_') for t in ticker_index if t in maxima_dict]
    dataframe_names_weighted = [t.replace('^', '').replace('.', '_') for t in [
        '^NSEI', '^NSEBANK', 'AXISBANK.NS', 'BHARTIARTL.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
        'INFY.NS', 'ITC.NS', 'RELIANCE.NS', 'SBIN.NS', 'TCS.NS', 'LT.NS'
    ] if t in maxima_dict]
    dataframe_names_index_weighted = [t.replace('^', '').replace('.', '_') for t in [
        '^CNXAUTO', '^CNXFMCG', '^CNXENERGY', '^CNXIT'
    ] if t in maxima_dict]

    merged_df = calculate_active_occurrences(dataframes, dataframe_names)
    if merged_df is None:
        return
    merged_df['close'] = maxima_dict.get('^NSEI', pd.DataFrame())['close'] if '^NSEI' in maxima_dict else pd.Series()
    Analysis_data = merged_df[['close', 'Short', 'Long']]
    Analysis_data_df = Analysis_data.sort_index(ascending=False)

    merged_df_weighted = calculate_active_occurrences(dataframes_weighted, dataframe_names_weighted)
    if merged_df_weighted is None:
        return
    merged_df_weighted['close'] = maxima_dict.get('^NSEI', pd.DataFrame())['close'] if '^NSEI' in maxima_dict else pd.Series()
    Analysis_data_weighted = merged_df_weighted[['close', 'Short', 'Long']]
    Analysis_data_weighted = Analysis_data_weighted.sort_index(ascending=False)

    merged_df_index = calculate_active_occurrences(dataframes_index, dataframe_names_index)
    if merged_df_index is None:
        return
    merged_df_index['close'] = maxima_dict.get('^NSEI', pd.DataFrame())['close'] if '^NSEI' in maxima_dict else pd.Series()
    Analysis_data_index = merged_df_index[['close', 'Short', 'Long']]
    Analysis_data_index = Analysis_data_index.sort_index(ascending=False)

    merged_df_weighted_index = calculate_active_occurrences(dataframes_index_weighted, dataframe_names_index_weighted)
    if merged_df_weighted_index is None:
        return
    merged_df_weighted_index['close'] = maxima_dict.get('^NSEI', pd.DataFrame())['close'] if '^NSEI' in maxima_dict else pd.Series()
    Analysis_data_index_weighted = merged_df_weighted_index[['close', 'Short', 'Long']]
    Analysis_data_index_weighted = Analysis_data_index_weighted.sort_index(ascending=False)

    Analysis_data_weighted = Analysis_data_weighted.rename(columns={'Long': 'Long_W', 'Short': 'Short_W'})
    Analysis_data_weighted.drop(columns=['close'], inplace=True)
    Analysis_data_index = Analysis_data_index.rename(columns={'Long': 'Long_I', 'Short': 'Short_I'})
    Analysis_data_index.drop(columns=['close'], inplace=True)
    Analysis_data_index_weighted = Analysis_data_index_weighted.rename(columns={'Long': 'Long_I_W', 'Short': 'Short_I_W'})
    Analysis_data_index_weighted.drop(columns=['close'], inplace=True)

    combined_df = pd.concat([Analysis_data, Analysis_data_weighted, Analysis_data_index, Analysis_data_index_weighted], axis=1)
    combined_df = combined_df.sort_index(ascending=False)

    # Format numerical columns to integers or 2 decimal places
    combined_df['close'] = combined_df['close'].round(2)
    combined_df['Short'] = combined_df['Short'].astype(int)
    combined_df['Long'] = combined_df['Long'].astype(int)
    combined_df['Short_W'] = combined_df['Short_W'].astype(int)
    combined_df['Long_W'] = combined_df['Long_W'].astype(int)
    combined_df['Short_I'] = combined_df['Short_I'].astype(int)
    combined_df['Long_I'] = combined_df['Long_I'].astype(int)
    combined_df['Short_I_W'] = combined_df['Short_I_W'].astype(int)
    combined_df['Long_I_W'] = combined_df['Long_I_W'].astype(int)

    def apply_color_based_on_previous_value(val):
        if pd.isna(val):
            return ''
        if val > apply_color_based_on_previous_value.previous_value:
            apply_color_based_on_previous_value.previous_value = val
            return 'background-color: green'
        else:
            apply_color_based_on_previous_value.previous_value = val
            return 'background-color: red'
    apply_color_based_on_previous_value.previous_value = combined_df['close'].iloc[0] if not combined_df['close'].empty else 0

    def apply_color_long_i(val):
        if pd.isna(val):
            return ''
        if val >= 3:
            return 'background-color: red'
        else:
            return 'background-color: green'

    def apply_color_short_i(val):
        if pd.isna(val):
            return ''
        if val >= 3:
            return 'background-color: red'
        else:
            return 'background-color: green'

    def apply_color_long(val):
        if pd.isna(val):
            return ''
        if val > 9:
            return 'background-color: red'
        elif val <= 6:
            return 'background-color: green'
        else:
            return 'background-color: blue'

    def apply_color_short(val):
        if pd.isna(val):
            return ''
        if val >= 9:
            return 'background-color: red'
        elif 6 <= val <= 9:
            return 'background-color: blue'
        else:
            return 'background-color: green'

    def apply_color_long_w(val):
        if pd.isna(val):
            return ''
        if val >= 3:
            return 'background-color: red'
        else:
            return 'background-color: green'

    def apply_color_short_w(val):
        if pd.isna(val):
            return ''
        if val >= 3:
            return 'background-color: red'
        else:
            return 'background-color: green'

    def apply_color_long_w_i(val):
        if pd.isna(val):
            return ''
        if val >= 2:
            return 'background-color: red'
        else:
            return 'background-color: blue'

    def apply_color_short_w_i(val):
        if pd.isna(val):
            return ''
        if val >= 2:
            return 'background-color: red'
        else:
            return 'background-color: blue'

    styled_df = combined_df.style.applymap(apply_color_long_i, subset=['Long_I']) \
                                .applymap(apply_color_short_i, subset=['Short_I']) \
                                .applymap(apply_color_long, subset=['Long']) \
                                .applymap(apply_color_short, subset=['Short']) \
                                .applymap(apply_color_long_w, subset=['Long_W']) \
                                .applymap(apply_color_short_w, subset=['Short_W']) \
                                .applymap(apply_color_long_w_i, subset=['Long_I_W']) \
                                .applymap(apply_color_short_w_i, subset=['Short_I_W']) \
                                .apply(lambda x: x.apply(apply_color_based_on_previous_value), subset=['close']) \
                                .format({
                                    'close': '{:.2f}',
                                    'Short': '{:d}',
                                    'Long': '{:d}',
                                    'Short_W': '{:d}',
                                    'Long_W': '{:d}',
                                    'Short_I': '{:d}',
                                    'Long_I': '{:d}',
                                    'Short_I_W': '{:d}',
                                    'Long_I_W': '{:d}'
                                })

    st.markdown(f"<span class='calc-summary' style='color: red;'>General: {len(dataframes)} | Indices: {len(dataframes_index)} | Weighted: {len(dataframes_weighted)} | Index Weighted: {len(dataframes_index_weighted)}</span>", unsafe_allow_html=True)

    # Responsive dataframe with prioritized columns for mobile
    st.dataframe(
        data=styled_df,
        use_container_width=True,
        height=None,  # Auto height
        hide_index=False,
        column_order=['close', 'Short', 'Long', 'Short_I', 'Long_I', 'Short_W', 'Long_W', 'Short_I_W', 'Long_I_W'],
        column_config={
            'close': st.column_config.NumberColumn('Close', format="%.2f"),
            'Short': st.column_config.NumberColumn('Short', format="%d"),
            'Long': st.column_config.NumberColumn('Long', format="%d"),
            'Short_I': st.column_config.NumberColumn('Short_I', format="%d"),
            'Long_I': st.column_config.NumberColumn('Long_I', format="%d"),
            'Short_W': st.column_config.NumberColumn('Short_W', format="%d"),
            'Long_W': st.column_config.NumberColumn('Long_W', format="%d"),
            'Short_I_W': st.column_config.NumberColumn('Short_I_W', format="%d"),
            'Long_I_W': st.column_config.NumberColumn('Long_I_W', format="%d")
        }
    )

    rb_bounds = calc_rangebreak(maxima_dict['^NSEI'].index.to_series()) if '^NSEI' in maxima_dict else []
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    if '^NSEI' in maxima_dict:
        fig.add_trace(go.Candlestick(
                x=maxima_dict['^NSEI'].index,
                open=maxima_dict['^NSEI']['open'],
                high=maxima_dict['^NSEI']['high'],
                low=maxima_dict['^NSEI']['low'],
                close=maxima_dict['^NSEI']['close'],
                name='NSEI'), row=1, col=1)
        fig.add_trace(go.Scatter(
                x=maxima_dict['^NSEI'].index,
                y=Analysis_data['Short'],
                mode='lines',
                line=dict(color="red", width=2),
                name='Lower_high'), row=2, col=1)
        fig.add_trace(go.Scatter(
                x=maxima_dict['^NSEI'].index,
                y=Analysis_data['Long'],
                mode='lines',
                line=dict(color="green", width=2),
                name='Higher_high'), row=2, col=1)
        fig.add_trace(go.Scatter(
                x=maxima_dict['^NSEI'].index,
                y=[10] * len(maxima_dict['^NSEI']),
                mode='lines',
                line=dict(color="orange", width=2, dash='dash'),
                name='th_0'), row=2, col=1)
        fig.add_trace(go.Scatter(
                x=maxima_dict['^NSEI'].index,
                y=[15] * len(maxima_dict['^NSEI']),
                mode='lines',
                line=dict(color="red", width=2, dash='dash'),
                name='th_1'), row=2, col=1)
    fig.update_layout(
        title='Asset Data',
        xaxis_title='Time',
        yaxis_title='Live Levels',
        template='seaborn',
        xaxis_rangeslider_visible=False,
        height=600,  # Reduced for mobile
        margin=dict(l=10, r=10, t=50, b=50),
        showlegend=True,
        font=dict(size=10),  # Smaller font for mobile
        uirevision='constant'  # Preserve zoom/pan state
    )
    fig.update_xaxes(rangebreaks=rb_bounds)
    st.plotly_chart(fig, use_container_width=True)

# Main interface
st.title("Financial Analysis Platform")
st.markdown("Analyze market data with real-time insights.")

# Input form for better mobile UX
with st.form(key='analysis_form'):
    intervals = ['5m', '15m', '30m', '1h', '1d']
    interval = st.selectbox("Interval", intervals, index=intervals.index('5m'))
    start_dates, end_dates = get_date_range()
    # Set default start date to 5 business days before today
    default_start_date = (datetime.today() - BusinessDay(n=5)).date()
    start_date = st.date_input("Start Date", default_start_date, min_value=datetime.today() - timedelta(days=7), max_value=datetime.today())
    end_date = st.date_input("End Date", datetime.today(), min_value=datetime.today() - timedelta(days=7), max_value=datetime.today())
    auto_run = st.checkbox("Auto-Run (every 2 minutes)", value=True)
    submit_button = st.form_submit_button("Analyse")

# Define ticker lists
ticker = ['^NSEI', '^NSEBANK', 'ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 
          'BAJAJ-AUTO.NS', 'BAJAJFINSV.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'BPCL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 
          'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 
          'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'INFY.NS', 
          'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'LTIM.NS', 'M&M.NS', 'MARUTI.NS', 'NESTLEIND.NS', 'NTPC.NS', 
          'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS', 'SHRIRAMFIN.NS', 'SUNPHARMA.NS', 
          'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TCS.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS']

ticker_index = ['^INDIAVIX', '^CNXAUTO', '^CNXPSUBANK', '^NSEBANK', '^CNXFMCG', '^CNXMETAL', '^CNXPSE', '^CNXENERGY', 
                '^CNXPHARMA', '^CNXMNC', '^CNXIT', '^CNXINFRA', '^CNXCONSUM', '^CNXSERVICE', '^CNXCMDT', '^CNXREALTY', 
                '^CNXMEDIA', '^CNXFIN']

# Handle form submission
if submit_button:
    with st.spinner("Running analysis..."):
        start_time = time.time()
        run_analysis(ticker, ticker_index, interval, start_date, end_date)
        st.write(f"Analysis completed in {time.time() - start_time:.2f} seconds.")

# Auto-run logic
if auto_run:
    # Get current time in IST
    ist = pytz.timezone('Asia/Kolkata')
    current_time = datetime.now(ist)
    current_hour = current_time.hour
    current_minute = current_time.minute
    current_time_of_day = current_hour * 60 + current_minute  # Minutes since midnight
    start_time_of_day = 9 * 60  # 9:00 AM IST
    end_time_of_day = 15 * 60 + 40  # 3:40 PM IST

    if start_time_of_day <= current_time_of_day <= end_time_of_day:
        # Within trading hours, proceed with auto-run
        if 'last_run_time' not in st.session_state:
            st.session_state.last_run_time = 0
            st.session_state.run_count = 0
            st.session_state.start_time = time.time()

        # Check if 2 minutes have passed since last run or it's the first run
        current_time = time.time()
        if current_time - st.session_state.last_run_time >= 120:  # 2 minutes
            st.session_state.run_count += 1
            st.write(f"Auto-Run #{st.session_state.run_count} started at {datetime.now(ist).strftime('%H:%M:%S')} IST")
            with st.spinner("Running auto-analysis..."):
                start_time = time.time()
                run_analysis(ticker, ticker_index, interval, start_date, end_date)
                st.session_state.last_run_time = current_time
                st.write(f"Auto-Run #{st.session_state.run_count} completed in {time.time() - start_time:.2f} seconds.")

            # Check if total execution time is within 2.5 minutes (150 seconds)
            if time.time() - st.session_state.start_time >= 150:
                st.warning("Stopping auto-run to comply with Streamlit's 3-minute timeout.")
                st.session_state.last_run_time = 0
                st.session_state.run_count = 0
                st.session_state.start_time = time.time()
            else:
                # Wait for the remainder of the 2-minute interval
                time_to_wait = 120 - (time.time() - st.session_state.last_run_time)
                if time_to_wait > 0:
                    time.sleep(time_to_wait)
                st.rerun()  # Refresh the page to trigger the next run
    else:
        # Outside trading hours, disable auto-run
        st.info(f"Auto-Run is disabled outside trading hours (9:00 AM to 3:40 PM IST). Current time: {current_time.strftime('%H:%M:%S')} IST.")