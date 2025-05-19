# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:33:34 2024

@author: kuldeep.rana
"""

import time
import math
import warnings
import yaml
import pandas    as pd
import numpy     as np
import streamlit as st

import streamlit_authenticator as stauth
import plotly.graph_objects    as go

from datetime        import datetime, timedelta
from plotly.subplots import make_subplots    
from scipy.signal    import argrelextrema
from collections     import deque
from yaml.loader     import SafeLoader
from breeze_connect  import BreezeConnect
from json_reader     import read_config
from close_value     import CLV

def get_api_data_index_opti_v2(base_strike_price,right,interval,product_type,exchange_code,stock_code,expiry_date,session_token,from_date,to_date):        
    breeze                    = BreezeConnect(api_key="W57^039330`163143`w385Ug8404ORL1")
    breeze.generate_session(api_secret="y45173e22303284_=y`5IG9444@64S57", session_token=session_token)
    historical_data = [] 
    data_1m         = []
 
    strike_price_range        = range((base_strike_price-200), (base_strike_price+201), 50)    
    for i, strike_price in enumerate(strike_price_range):
        print(f"Set-Point:{i}:",strike_price)
        df_his = breeze.get_historical_data_v2(
                                            interval=interval,from_date=str(from_date), 
                                            to_date=str(to_date),stock_code=stock_code,
                                            exchange_code=exchange_code,product_type=product_type,
                                            expiry_date=str(expiry_date),right=right, strike_price=str(strike_price))
        if 'Success' in df_his:
            df_his = pd.DataFrame(df_his['Success'])
            df_his.columns = [f'{col}_sp{i+1}' for col in df_his.columns]  
            historical_data.append(df_his)
    data_1m = pd.concat(historical_data, axis=1, ignore_index=False)
    return data_1m
    
def get_api_data_index_fut_v2(right,interval,product_type,exchange_code,stock_code,expiry_date,session_token,from_date,to_date):        
    breeze                    = BreezeConnect(api_key="W57^039330`163143`w385Ug8404ORL1")
    breeze.generate_session(api_secret="y45173e22303284_=y`5IG9444@64S57", session_token=session_token)

    historical_fut  = []
   
    df_his_fut = breeze.get_historical_data_v2(
                                        interval=interval,from_date=str(from_date), 
                                        to_date=str(to_date),stock_code=stock_code,
                                        exchange_code=exchange_code,
                                        product_type="futures",right="others",strike_price="0",
                                        expiry_date=str(expiry_date))
    print("Getting-Futures-Data ......:")
    historical_fut  = df_his_fut['Success']       
    historical_fut  = pd.DataFrame(historical_fut) 
    # print(historical_fut)       
    print("Done") 
    return historical_fut
def get_date_range():
    today = datetime.today()
    start_dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(61)]  # Past 2 months
    end_dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(4)]  # Next 2 days
    return start_dates, end_dates

exchange = 'NF'
start_date       = "2024-05-15"
end_date         =  "2024-05-20"
expiry_date     = "2024-05-23"
expiry_date_fut = "2024-05-30"
ticker = 'NIFTY'

start_time = time.time()    


config = read_config(config_file="config.json", readjson_with_comments=True) 
session_token = config["session_token"]
open_price_nf = config["open_nf"]
current_strike_price = CLV.find_idx_nearest_val(open_price_nf)   
current_strike_price      

df_opti = get_api_data_index_opti_v2(current_strike_price,config["right"],config["interval"],config["product_type"],config["exchange_code"],config["stock_code"],str(expiry_date)+"T07:00:00.000Z",config["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       
df_opti_pe = get_api_data_index_opti_v2(current_strike_price,config["right_c"],config["interval"],config["product_type"],config["exchange_code"],config["stock_code"],str(expiry_date)+"T07:00:00.000Z",config["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       

sp1 = df_opti[["close_sp1","datetime_sp1"]]
sp2 = df_opti[['close_sp2',"datetime_sp1"]]
sp3 = df_opti[['close_sp3',"datetime_sp1"]]
sp4 = df_opti[['close_sp4',"datetime_sp1"]]
sp5 = df_opti[['close_sp5',"datetime_sp1"]]
sp6 = df_opti[['close_sp6',"datetime_sp1"]]
sp7 = df_opti[['close_sp7',"datetime_sp1"]]
sp8 = df_opti[['close_sp8',"datetime_sp1"]]
sp9 = df_opti[['close_sp9',"datetime_sp1"]]

sp1.rename(columns={'close_sp1': 'Close'}, inplace=True)
sp2.rename(columns={'close_sp2': 'Close'}, inplace=True)
sp3.rename(columns={'close_sp3': 'Close'}, inplace=True)
sp4.rename(columns={'close_sp4': 'Close'}, inplace=True)
sp5.rename(columns={'close_sp5': 'Close'}, inplace=True)
sp6.rename(columns={'close_sp6': 'Close'}, inplace=True)
sp7.rename(columns={'close_sp7': 'Close'}, inplace=True)
sp8.rename(columns={'close_sp8': 'Close'}, inplace=True)
sp9.rename(columns={'close_sp9': 'Close'}, inplace=True)


sp1['datetime_column'] = pd.to_datetime(sp1['datetime_sp1'])
sp1.set_index('datetime_column', inplace=True)

sp2['datetime_column'] = pd.to_datetime(sp2['datetime_sp1'])
sp2.set_index('datetime_column', inplace=True)

sp3['datetime_column'] = pd.to_datetime(sp3['datetime_sp1'])
sp3.set_index('datetime_column', inplace=True)

sp4['datetime_column'] = pd.to_datetime(sp4['datetime_sp1'])
sp4.set_index('datetime_column', inplace=True)

sp5['datetime_column'] = pd.to_datetime(sp5['datetime_sp1'])
sp5.set_index('datetime_column', inplace=True)

sp6['datetime_column'] = pd.to_datetime(sp6['datetime_sp1'])
sp6.set_index('datetime_column', inplace=True)

sp7['datetime_column'] = pd.to_datetime(sp7['datetime_sp1'])
sp7.set_index('datetime_column', inplace=True)

sp8['datetime_column'] = pd.to_datetime(sp8['datetime_sp1'])
sp8.set_index('datetime_column', inplace=True)
      
sp9['datetime_column'] = pd.to_datetime(sp9['datetime_sp1'])
sp9.set_index('datetime_column', inplace=True)
      

def cal_data(data):
    global highs
    
    data['Close'] = data['Close'].apply(pd.to_numeric, errors='coerce')

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
    # data_dates = data.datetime_sp1
    data['datetime_column'] = pd.to_datetime(data['datetime_sp1'])
    data.set_index('datetime_column', inplace=True)
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
        # if df.index[i] in out_hh_:
        #     label.append('HH')
        if df.index[i] in out_hl_:
            label.append('HL')
        # elif df.index[i] in out_ll_:
        #     label.append('LL')
        elif df.index[i] in out_lh_:
            label.append('LH')
        else:
            label.append(None)

    df['label'] = label
    return df


sp1.dtypes
 
sp1_maxima = cal_data(sp1)       
sp2_maxima = cal_data(sp2)
sp3_maxima = cal_data(sp3)
sp4_maxima = cal_data(sp4)
sp5_maxima = cal_data(sp5)
sp6_maxima = cal_data(sp6)
sp7_maxima = cal_data(sp7)
sp8_maxima = cal_data(sp8)
sp9_maxima = cal_data(sp9)


dataframes = [sp1_maxima,sp2_maxima,sp3_maxima,sp4_maxima,sp5_maxima,sp6_maxima,sp7_maxima,sp8_maxima,sp9_maxima,]
    
def rename_and_extract_label(df, name):
    df = df.rename(columns={'label': f'{name}_label'})
    return df[[f'{name}_label']]

def calculate_active_occurrences(dataframes):
    renamed_dfs = []
    column_names = [df_name.split('_')[0] for df_name in globals().keys() if 'maxima' in df_name]
    
    for df, name in zip(dataframes, column_names):
        renamed_df = rename_and_extract_label(df, name)
        renamed_dfs.append(renamed_df)

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

def calc_rangebreak(time_series:pd.Series):           
    timedeltas = time_series.diff()            
    if len(time_series) < 2:
        return []            
    missing_times = np.where([timedeltas > timedeltas.median()*1.5])[1]
    off = pd.Timedelta(seconds=0.0001)
    rb = [{'bounds': [str((time_series.iloc[t-1]+off)), str((time_series.iloc[t]-off))]} for t in missing_times]
    return rb
close_column=df_opti["close_sp1"].values
merged_df = calculate_active_occurrences(dataframes)
merged_df["close"] =  close_column
Analysis_data = merged_df[['close', 'Short', 'Long']]
Analysis_data_df = Analysis_data.sort_index(ascending=False)
            

def apply_color_based_on_previous_value(val):
    if val > apply_color_based_on_previous_value.previous_value:
        apply_color_based_on_previous_value.previous_value = val
        return 'background-color: green'
    else:
        apply_color_based_on_previous_value.previous_value = val
        return 'background-color: red'    
apply_color_based_on_previous_value.previous_value = Analysis_data_df['close'].iloc[0]      
      
def apply_color_long_i(val):
    if val >= 3:
        return 'background-color: red'
    elif val <= 3:
        return 'background-color: green'
    else:
        return ''
def apply_color_short_i(val):
    if val >= 3:
        return 'background-color: red'
    elif val <= 3:
        return 'background-color: green'
    else:
        return ''        
def apply_color_long(val):
    if val > 9:
        return 'background-color: red'
    elif val <= 6:
        return 'background-color: green'
    if 6 >= val <= 9:
        return 'background-color: blue'
    else:
        return ''        
def apply_color_short(val):
    if val >= 9:
        return 'background-color: red'
    if 6 >= val <= 9:
        return 'background-color: blue'
    elif val <= 6:
        return 'background-color: green'
    else:
        return ''            
def apply_color_long_w(val):
    if val >= 3:
        return 'background-color: red'
    elif val <= 3:
        return 'background-color: green'
    else:
        return ''        
def apply_color_short_w(val):
    if val >= 3:
        return 'background-color: red'
    elif val <= 3:
        return 'background-color: green'
    else:
        return ''
def apply_color_long_w_i(val):
    if val >= 2:
        return 'background-color: red'
    elif val <= 2:
        return 'background-color: blue'
    else:
        return ''        
def apply_color_short_w_i(val):
    if val >= 2:
        return 'background-color: red'
    elif val <= 2:
        return 'background-color: blue'
    else:
        return ''
def apply_color_to_index(index):
    return 'background-color: orange'

styled_df = Analysis_data_df.style.applymap(apply_color_long_i, subset=['Long']) \
                              .applymap(apply_color_short_i, subset=['Short']) \
                              .apply(lambda x: x.apply(apply_color_based_on_previous_value), subset=['close']) 
                              
# st.write(f"<span style='color: red; font-size: 18px;'>General calculations : {len(dataframes)} : Indices calculations {len(dataframes_index)} : Weighted calculations: {len(dataframes_weighted)} :  Index Weighted calculations: {len(dataframes_index_weighted)}</span>", unsafe_allow_html=True)

st.dataframe(data=styled_df, width=1200, height=600, use_container_width=False, hide_index=False, column_order=['close','Short',"Long"], column_config=None)
        
# rb_bounds   = calc_rangebreak(NSEI.index.to_series())
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,shared_yaxes=True,vertical_spacing=0.05,row_heights=[0.3, 0.1])
# fig.add_trace(go.Candlestick(
#         x=NSEI.index,
#         open=NSEI['open'],
#         high=NSEI['high'],
#         low=NSEI['low'],
#         close=NSEI['close'],
#         name='NSEI'), row=1, col=1 )            
# fig.add_trace(go.Scatter(
#         x=NSEI.index,
#         y=Analysis_data['Short'], 
#         mode='lines',
#         line=dict(color="red",width=3),
#         name='Lower_high'), row=2, col=1)      
# fig.add_trace(go.Scatter(
#         x=NSEI.index,
#         y=Analysis_data['Long'], 
#         mode='lines',
#         line=dict(color="green",width=3),
#         name='Higher_high'), row=2, col=1)   
# fig.add_trace(go.Scatter(
#         x=NSEI.index,
#         y=[10] * len(NSEI), 
#         mode='lines',
#         line=dict(color="orange",width=3, dash='dash'),
#         name='th_0'), row=2, col=1)
# fig.add_trace(go.Scatter(
#         x=NSEI.index,
#         y=[15] * len(NSEI), 
#         mode='lines',
#         line=dict(color="red",width=3, dash='dash'),
#         name='th_1'), row=2, col=1)            
# fig.update_layout(
#         title='Asset Data',
#         xaxis_title='time',
#         yaxis_title='Live_levels',
#         template='seaborn' ,
#         xaxis_rangeslider_visible=False,
#         width=1200,height=1600)
# (fig.update_xaxes(rangebreaks=rb_bounds))
# st.plotly_chart(fig)
# st.set_option('deprecation.showPyplotGlobalUse', False)

