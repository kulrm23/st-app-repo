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
# from close_value     import CLV

from streamlit_authenticator.utilities.exceptions import (CredentialsError,ForgotError,LoginError,RegisterError,ResetError,UpdateError) 

warnings.filterwarnings('ignore')
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)
st.set_page_config(layout="wide")
authenticator = stauth.Authenticate(config['credentials'],config['cookie']['name'], config['cookie']['key'],config['cookie']['expiry_days'],config['pre-authorized'])
authenticator.login()
if st.session_state["authentication_status"]:
    authenticator.logout(location='sidebar')
    st.write(f'Welcome to AI ML *{st.session_state["name"]}*') 
    @st.cache_resource
    @st.cache_data
    @st.cache(ttl=1*600)
    
    def get_api_data_index_opti_v2(strike_price,right,interval,product_type,exchange_code,stock_code,expiry_date,session_token,from_date,to_date):        
  
        df_his = breeze.get_historical_data_v2(
                                            interval=interval,from_date=str(from_date), 
                                            to_date=str(to_date),stock_code=stock_code,
                                            exchange_code=exchange_code,product_type=product_type,
                                            expiry_date=str(expiry_date),right=right, strike_price=str(strike_price))
        if 'Success' in df_his:
            df_his = pd.DataFrame(df_his['Success'])
        return df_his
        
    def get_api_data_index_fut_v2(right,interval,product_type,exchange_code,stock_code,expiry_date,session_token,from_date,to_date):        

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
        print("Done") 
        return historical_fut
    def get_date_range():
        today = datetime.today()
        start_dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(61)]  # Past 2 months
        end_dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(4)]  # Next 2 days
        return start_dates, end_dates
    
    exchange = 'NF'
    exchange1= "NFO"
    ticker_list = ['CNXBAN']
    intervals = ['5minute','1minute']
    interval = st.selectbox("Select Interval", intervals)
    start_dates, end_dates = get_date_range()
    start_date = st.date_input("Start Date", datetime.today() - timedelta(days=3))
    end_date = st.date_input("End Date", datetime.today() + timedelta(days=1))
    expiry_date = st.date_input("Expiry Date", datetime.today())
    expiry_date_fut = st.date_input("Expiry Date Futures", datetime.today())

    ticker = st.selectbox("Select ticker", ticker_list)
    list_stike = [48000,48100,48200,48300,48400,48500,48600,48700,48800,48900,49000,49100,49200,49300,49400]
    strike_price= st.selectbox("Enter ATM strike", list_stike)
    start_time = time.time()    
    
    if st.button("Analyse"):
        config_icic = read_config(config_file="config.json", readjson_with_comments=True) 
        session_token = config_icic["session_token"]
        # open_price_nf = config_icic["open_nf"]
        breeze                    = BreezeConnect(api_key="W57^039330`163143`w385Ug8404ORL1")
        breeze.generate_session(api_secret="y45173e22303284_=y`5IG9444@64S57", session_token=session_token)
        
        sp1 = get_api_data_index_opti_v2(strike_price-400,config_icic["right"],interval,config_icic["product_type"],exchange1,ticker,str(expiry_date)+"T07:00:00.000Z",config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       
        sp2 = get_api_data_index_opti_v2(strike_price-300,config_icic["right"],interval,config_icic["product_type"],exchange1,ticker,str(expiry_date)+"T07:00:00.000Z",config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       
        sp3 = get_api_data_index_opti_v2(strike_price-200,config_icic["right"],interval,config_icic["product_type"],exchange1,ticker,str(expiry_date)+"T07:00:00.000Z",config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       
        sp4 = get_api_data_index_opti_v2(strike_price-100,config_icic["right"],interval,config_icic["product_type"],exchange1,ticker,str(expiry_date)+"T07:00:00.000Z",config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       
        sp5 = get_api_data_index_opti_v2(strike_price+100,config_icic["right"],interval,config_icic["product_type"],exchange1,ticker,str(expiry_date)+"T07:00:00.000Z",config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       
        sp6 = get_api_data_index_opti_v2(strike_price+200,config_icic["right"],interval,config_icic["product_type"],exchange1,ticker,str(expiry_date)+"T07:00:00.000Z",config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       
        sp7 = get_api_data_index_opti_v2(strike_price+300,config_icic["right"],interval,config_icic["product_type"],exchange1,ticker,str(expiry_date)+"T07:00:00.000Z",config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       
        sp8 = get_api_data_index_opti_v2(strike_price+400,config_icic["right"],interval,config_icic["product_type"],exchange1,ticker,str(expiry_date)+"T07:00:00.000Z",config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       
        sp9 = get_api_data_index_opti_v2(strike_price,config_icic["right"],interval,config_icic["product_type"],exchange1,ticker,str(expiry_date)+"T07:00:00.000Z",config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       
        
        historical_fut = get_api_data_index_fut_v2(config_icic["right"],interval,config_icic["product_type_fut"],exchange1,ticker,expiry_date_fut,config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")
        # historical_fut
        # sp1
        historical_fut.rename(columns={'close': 'Close'}, inplace=True)
        historical_fut['datetime_column'] = pd.to_datetime(historical_fut['datetime'])
        historical_fut.set_index('datetime_column', inplace=True)
        
        sp1.rename(columns={'close': 'Close'}, inplace=True)
        sp2.rename(columns={'close': 'Close'}, inplace=True)
        sp3.rename(columns={'close': 'Close'}, inplace=True)
        sp4.rename(columns={'close': 'Close'}, inplace=True)
        sp5.rename(columns={'close': 'Close'}, inplace=True)
        sp6.rename(columns={'close': 'Close'}, inplace=True)
        sp7.rename(columns={'close': 'Close'}, inplace=True)
        sp8.rename(columns={'close': 'Close'}, inplace=True)
        sp9.rename(columns={'close': 'Close'}, inplace=True)
        
        sp1['datetime_column'] = pd.to_datetime(sp1['datetime'])
        sp1.set_index('datetime_column', inplace=True)
        
        sp2['datetime_column'] = pd.to_datetime(sp2['datetime'])
        sp2.set_index('datetime_column', inplace=True)
        
        sp3['datetime_column'] = pd.to_datetime(sp3['datetime'])
        sp3.set_index('datetime_column', inplace=True)
        
        sp4['datetime_column'] = pd.to_datetime(sp4['datetime'])
        sp4.set_index('datetime_column', inplace=True)
        
        sp5['datetime_column'] = pd.to_datetime(sp5['datetime'])
        sp5.set_index('datetime_column', inplace=True)
        
        sp6['datetime_column'] = pd.to_datetime(sp6['datetime'])
        sp6.set_index('datetime_column', inplace=True)
        
        sp7['datetime_column'] = pd.to_datetime(sp7['datetime'])
        sp7.set_index('datetime_column', inplace=True)
        
        sp8['datetime_column'] = pd.to_datetime(sp8['datetime'])
        sp8.set_index('datetime_column', inplace=True)
              
        sp9['datetime_column'] = pd.to_datetime(sp9['datetime'])
        sp9.set_index('datetime_column', inplace=True)
        
        
        sp1_ce = get_api_data_index_opti_v2(strike_price-400,config_icic["right_c"],interval,config_icic["product_type"],exchange1,ticker,str(expiry_date)+"T07:00:00.000Z",config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       
        sp2_ce = get_api_data_index_opti_v2(strike_price-300,config_icic["right_c"],interval,config_icic["product_type"],exchange1,ticker,str(expiry_date)+"T07:00:00.000Z",config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       
        sp3_ce = get_api_data_index_opti_v2(strike_price-200,config_icic["right_c"],interval,config_icic["product_type"],exchange1,ticker,str(expiry_date)+"T07:00:00.000Z",config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       
        sp4_ce = get_api_data_index_opti_v2(strike_price-100,config_icic["right_c"],interval,config_icic["product_type"],exchange1,ticker,str(expiry_date)+"T07:00:00.000Z",config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       
        sp5_ce = get_api_data_index_opti_v2(strike_price+100,config_icic["right_c"],interval,config_icic["product_type"],exchange1,ticker,str(expiry_date)+"T07:00:00.000Z",config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       
        sp6_ce = get_api_data_index_opti_v2(strike_price+200,config_icic["right_c"],interval,config_icic["product_type"],exchange1,ticker,str(expiry_date)+"T07:00:00.000Z",config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       
        sp7_ce = get_api_data_index_opti_v2(strike_price+300,config_icic["right_c"],interval,config_icic["product_type"],exchange1,ticker,str(expiry_date)+"T07:00:00.000Z",config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       
        sp8_ce = get_api_data_index_opti_v2(strike_price+400,config_icic["right_c"],interval,config_icic["product_type"],exchange1,ticker,str(expiry_date)+"T07:00:00.000Z",config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       
        sp9_ce = get_api_data_index_opti_v2(strike_price,config_icic["right_c"],interval,config_icic["product_type"],exchange1,ticker,str(expiry_date)+"T07:00:00.000Z",config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       
        
        sp1_ce.rename(columns={'close': 'Close'}, inplace=True)
        sp2_ce.rename(columns={'close': 'Close'}, inplace=True)
        sp3_ce.rename(columns={'close': 'Close'}, inplace=True)
        sp4_ce.rename(columns={'close': 'Close'}, inplace=True)
        sp5_ce.rename(columns={'close': 'Close'}, inplace=True)
        sp6_ce.rename(columns={'close': 'Close'}, inplace=True)
        sp7_ce.rename(columns={'close': 'Close'}, inplace=True)
        sp8_ce.rename(columns={'close': 'Close'}, inplace=True)
        sp9_ce.rename(columns={'close': 'Close'}, inplace=True)
        
        
        sp1_ce['datetime_column'] = pd.to_datetime(sp1_ce['datetime'])
        sp1_ce.set_index('datetime_column', inplace=True)
        
        sp2_ce['datetime_column'] = pd.to_datetime(sp2_ce['datetime'])
        sp2_ce.set_index('datetime_column', inplace=True)
        
        sp3_ce['datetime_column'] = pd.to_datetime(sp3_ce['datetime'])
        sp3_ce.set_index('datetime_column', inplace=True)
        
        sp4_ce['datetime_column'] = pd.to_datetime(sp4_ce['datetime'])
        sp4_ce.set_index('datetime_column', inplace=True)
        
        sp5_ce['datetime_column'] = pd.to_datetime(sp5_ce['datetime'])
        sp5_ce.set_index('datetime_column', inplace=True)
        
        sp6_ce['datetime_column'] = pd.to_datetime(sp6_ce['datetime'])
        sp6_ce.set_index('datetime_column', inplace=True)
        
        sp7_ce['datetime_column'] = pd.to_datetime(sp7_ce['datetime'])
        sp7_ce.set_index('datetime_column', inplace=True)
        
        sp8_ce['datetime_column'] = pd.to_datetime(sp8_ce['datetime'])
        sp8_ce.set_index('datetime_column', inplace=True)
              
        sp9_ce['datetime_column'] = pd.to_datetime(sp9_ce['datetime'])
        sp9_ce.set_index('datetime_column', inplace=True)       

        def cal_data(data):
            start_time = time.time()
            
            data['Close'] = data['Close'].apply(pd.to_numeric, errors='coerce')
        
            order = 5
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
            data['datetime_column'] = pd.to_datetime(data['datetime'])
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
         
        sp1_maxima = cal_data(sp1)       
        sp2_maxima = cal_data(sp2)
        sp3_maxima = cal_data(sp3)
        sp4_maxima = cal_data(sp4)
        sp5_maxima = cal_data(sp5)
        sp6_maxima = cal_data(sp6)
        sp7_maxima = cal_data(sp7)
        sp8_maxima = cal_data(sp8)
        sp9_maxima = cal_data(sp9)
        
        sp1_ce_maxima = cal_data(sp1_ce)       
        sp2_ce_maxima = cal_data(sp2_ce)
        sp3_ce_maxima = cal_data(sp3_ce)
        sp4_ce_maxima = cal_data(sp4_ce)
        sp5_ce_maxima = cal_data(sp5_ce)
        sp6_ce_maxima = cal_data(sp6_ce)
        sp7_ce_maxima = cal_data(sp7_ce)
        sp8_ce_maxima = cal_data(sp8_ce)
        sp9_ce_maxima = cal_data(sp9_ce)
        
        historical_fut_maxima = cal_data(historical_fut)       
                
        dataframes = [sp1_maxima,sp2_maxima,sp3_maxima,sp4_maxima,sp5_maxima,sp6_maxima,sp7_maxima,sp8_maxima,sp9_maxima,]
        dataframes_ce = [sp1_ce_maxima,sp2_ce_maxima,sp3_ce_maxima,sp4_ce_maxima,sp5_ce_maxima,sp6_ce_maxima,sp7_ce_maxima,sp8_ce_maxima,sp9_ce_maxima,]
        dataframes_fut = [historical_fut_maxima]
        
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
        close_column_fut = historical_fut["close"].values
        close_column = sp5["close"].values
        merged_df = calculate_active_occurrences(dataframes)
        # merged_df["close"] =  close_column_fut
        Analysis_data = merged_df[[ 'Short', 'Long']]
        Analysis_data = Analysis_data.rename(columns={'Long': 'Long_pe','Short': 'Short_pe'}) 
        Analysis_data_df = Analysis_data.sort_index(ascending=False)
        
        close_column_ce = sp1_ce["close"].values
        merged_df_ce = calculate_active_occurrences(dataframes_ce)
        # merged_df_ce["close"] =  close_column_fut
        Analysis_data_ce = merged_df_ce[['Short', 'Long']]
        Analysis_data_ce = Analysis_data_ce.rename(columns={'Long': 'Long_ce','Short': 'Short_ce'}) 
        Analysis_data_df_ce = Analysis_data_ce.sort_index(ascending=False)
        
        
        merged_df_fut = calculate_active_occurrences(dataframes_fut)
        # merged_df_fut["close"] =  close_column_fut
        Analysis_data_fut = merged_df_fut[['Short', 'Long']]
        Analysis_data_fut = Analysis_data_fut.rename(columns={'Long': 'Long_fut','Short': 'Short_fut'}) 
        Analysis_data_df_fut = Analysis_data_fut.sort_index(ascending=False)
        
        combined_df = pd.concat([Analysis_data_df,Analysis_data_df_ce,Analysis_data_df_fut], axis=1)  
        combined_df = combined_df.sort_index(ascending=False)

        
        
        def apply_color_based_on_previous_value(val):
            if val > apply_color_based_on_previous_value.previous_value:
                apply_color_based_on_previous_value.previous_value = val
                return 'background-color: green'
            else:
                apply_color_based_on_previous_value.previous_value = val
                return 'background-color: red'    
        # apply_color_based_on_previous_value.previous_value = combined_df['close_pe'].iloc[0]      
              
        def apply_color_long_i(val):
            if val >= 3:
                return 'background-color: green'
            elif val <= 3:
                return 'background-color: blue'
            else:
                return ''
        def apply_color_short_i(val):
            if val >= 3:
                return 'background-color: red'
            elif val <= 3:
                return 'background-color: blue'
            else:
                return ''        
        def apply_color_long(val):
            if val > 0:
                return 'background-color: green'
            elif val < 1:
                return 'background-color: blue'
            else:
                return ''        
        def apply_color_short(val):
            if val > 0:
                return 'background-color: red'
            elif val < 1:
                return 'background-color: blue'
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
        
        styled_df = combined_df.style.applymap(apply_color_long_i, subset=['Long_pe']) \
                                      .applymap(apply_color_short_i, subset=['Short_pe']) \
                                      .applymap(apply_color_long, subset=['Long_ce']) \
                                      .applymap(apply_color_short, subset=['Short_ce']) \
                                      .applymap(apply_color_long, subset=['Long_fut']) \
                                      .applymap(apply_color_short, subset=['Short_fut']) \
                                      # .apply(lambda x: x.apply(apply_color_based_on_previous_value), subset=['close_fut'])\
                                      # .apply(lambda x: x.apply(apply_color_based_on_previous_value), subset=['close_pe'])\
                                      # .apply(lambda x: x.apply(apply_color_based_on_previous_value), subset=['close_ce']) 
                                      
        # st.write(f"<span style='color: red; font-size: 18px;'>General calculations : {len(dataframes)} : Indices calculations {len(dataframes_index)} : Weighted calculations: {len(dataframes_weighted)} :  Index Weighted calculations: {len(dataframes_index_weighted)}</span>", unsafe_allow_html=True)
        
        st.dataframe(data=styled_df, width=1600, height=600, use_container_width=False, hide_index=False, column_order=['Short_pe',"Long_ce",'Short_ce','Long_pe','Short_fut','Long_fut'], column_config=None)
                
        
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
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time
    
elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')

# """ Below code is for security: refer :https://github.com/mkhorasani/Streamlit-Authenticator/tree/main?tab=readme-ov-file#authenticateregister_user"""
st.sidebar.header("User Access Actions")

show_register_form = st.sidebar.checkbox("New Registration")
if show_register_form:
    try:         
        (email_of_registered_user,
            username_of_registered_user,
            name_of_registered_user) = authenticator.register_user(pre_authorization=False,location='sidebar')              
        if email_of_registered_user:
            st.success('User registered successfully')
    except RegisterError as e:
        st.error(e)
        print(e)
            
with open('config.yaml', 'w', encoding='utf-8') as file:
    yaml.dump(config, file, default_flow_style=False)     
           
show_forgotten_password_form = st.sidebar.checkbox("Forgot Password")
if show_forgotten_password_form:
    try:
        (username_of_forgotten_password,
            email_of_forgotten_password,
            new_random_password) = authenticator.forgot_password(location='sidebar')
        if username_of_forgotten_password:
            st.success('New password sent securely')
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
        except UpdateError as e:
            st.error(e)

show_Reset_password_form = st.sidebar.checkbox("Reset Password")
if show_Reset_password_form:
    if st.session_state["authentication_status"]:
        try:
            if authenticator.reset_password(st.session_state["username"]):
                st.success('Password modified successfully')
        except ResetError as e:
            st.error(e)
        except CredentialsError as e:
            st.error(e)       

with open('config.yaml', 'w', encoding='utf-8') as file:
    yaml.dump(config, file, default_flow_style=False)       


# from stream_lit_slack import lambda_handler
      
# if Analysis_data['Short'].iloc[-1] >= 6:
#    message1=f"Short_buildup and the Value is : {Analysis_data['Short'].iloc[-1]}" 
#    lambda_handler(True, message1)           

