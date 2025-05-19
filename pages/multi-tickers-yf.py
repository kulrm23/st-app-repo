# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:33:34 2024

@author: kuldeep.rana
"""

import time
import warnings
import yaml
import pandas    as pd
import yfinance  as yf
import numpy     as np
import streamlit as st

import streamlit_authenticator as stauth
import plotly.graph_objects    as go

from datetime        import datetime, timedelta
from plotly.subplots import make_subplots    
from scipy.signal    import argrelextrema
from collections     import deque
from yaml.loader     import SafeLoader

from streamlit_authenticator.utilities.exceptions import (CredentialsError,ForgotError,LoginError,RegisterError,ResetError,UpdateError) 

warnings.filterwarnings('ignore')
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(config['credentials'],config['cookie']['name'], config['cookie']['key'],config['cookie']['expiry_days'],config['pre-authorized'])
authenticator.login()

if st.session_state["authentication_status"]:
    authenticator.logout(location='sidebar')
    st.write(f'Welcome to AI ML *{st.session_state["name"]}*') 

    def get_data(ticker, interval, start, end):
        yfObj = yf.Ticker(ticker)
        data = yfObj.history(start=start, end=end, interval=interval)
        return data
    def get_date_range():
        today = datetime.today()
        start_dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(61)]  # Past 2 months
        end_dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(4)]  # Next 2 days
        return start_dates, end_dates
    
    intervals = ['5m', '15m', '30m', '1h', '1d', '5d', '1wk', '1mo', '3mo','1m', '2m']
    interval = st.selectbox("Select Interval", intervals)
    start_dates, end_dates = get_date_range()
    start_date = st.date_input("Start Date", datetime.today() - timedelta(days=3))
    end_date = st.date_input("End Date", datetime.today() + timedelta(days=1))
    
    start_time = time.time()
    
    ticker = ['^NSEI', 'GC=F', '^NSEBANK', 'ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJAJFINSV.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'BPCL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS',
              'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'INFY.NS', 'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'LTIM.NS', 'M&M.NS', 'MARUTI.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS', 'SHRIRAMFIN.NS', 'SUNPHARMA.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TCS.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS']
    
    ticker_index = ['^INDIAVIX','^CNXAUTO','^CNXPSUBANK','NIFTYPVTBANK.NS','^CNXFMCG','^CNXMETAL','^CNXPSE','^CNXENERGY','^CNXPHARMA','^CNXMNC','CPSE.NS','^CNXIT','^CNXINFRA','^CNXCONSUM','^CNXSERVICE','^CNXCMDT','^CNXREALTY','^CNXMEDIA','^CNXFIN']
   
    
    if st.button("Analyse"):
        
        NSEI = get_data(ticker[0], interval, start_date, end_date)
        gold = get_data(ticker[1], interval, start_date, end_date)
        NSEBANK = get_data(ticker[2], interval, start_date, end_date)
        ADANIENT = get_data(ticker[3], interval, start_date, end_date)
        ADANIPORTS = get_data(ticker[4], interval, start_date, end_date)
        APOLLOHOSP = get_data(ticker[5], interval, start_date, end_date)
        ASIANPAINT = get_data(ticker[6], interval, start_date, end_date)
        AXISBANK = get_data(ticker[7], interval, start_date, end_date)
        BAJAJ_AUTO = get_data(ticker[8], interval, start_date, end_date)
        BAJAJFINSV = get_data(ticker[9], interval, start_date, end_date)
        BAJFINANCE = get_data(ticker[10], interval, start_date, end_date)
        BHARTIARTL = get_data(ticker[11], interval, start_date, end_date)
        BPCL = get_data(ticker[12], interval, start_date, end_date)
        BRITANNIA = get_data(ticker[13], interval, start_date, end_date)
        CIPLA = get_data(ticker[14], interval, start_date, end_date)
        COALINDIA = get_data(ticker[15], interval, start_date, end_date)
        DIVISLAB = get_data(ticker[16], interval, start_date, end_date)
        DRREDDY = get_data(ticker[17], interval, start_date, end_date)
        EICHERMOT = get_data(ticker[18], interval, start_date, end_date)
        GRASIM = get_data(ticker[19], interval, start_date, end_date)
        HCLTECH = get_data(ticker[20], interval, start_date, end_date)
        HDFCBANK = get_data(ticker[21], interval, start_date, end_date)
        HDFCLIFE = get_data(ticker[22], interval, start_date, end_date)
        HEROMOTOCO = get_data(ticker[23], interval, start_date, end_date)
        HINDALCO = get_data(ticker[24], interval, start_date, end_date)
        HINDUNILVR = get_data(ticker[25], interval, start_date, end_date)
        ICICIBANK = get_data(ticker[26], interval, start_date, end_date)
        INDUSINDBK = get_data(ticker[27], interval, start_date, end_date)
        INFY = get_data(ticker[28], interval, start_date, end_date)
        ITC = get_data(ticker[29], interval, start_date, end_date)
        JSWSTEEL = get_data(ticker[30], interval, start_date, end_date)
        KOTAKBANK = get_data(ticker[31], interval, start_date, end_date)
        LT = get_data(ticker[32], interval, start_date, end_date)
        LTIM = get_data(ticker[33], interval, start_date, end_date)
        M_M = get_data(ticker[34], interval, start_date, end_date)
        MARUTI = get_data(ticker[35], interval, start_date, end_date)
        NESTLEIND = get_data(ticker[36], interval, start_date, end_date)
        NTPC = get_data(ticker[37], interval, start_date, end_date)
        ONGC = get_data(ticker[38], interval, start_date, end_date)
        POWERGRID = get_data(ticker[39], interval, start_date, end_date)
        RELIANCE = get_data(ticker[40], interval, start_date, end_date)
        SBILIFE = get_data(ticker[41], interval, start_date, end_date)
        SBIN = get_data(ticker[42], interval, start_date, end_date)
        SHRIRAMFIN = get_data(ticker[43], interval, start_date, end_date)
        SUNPHARMA = get_data(ticker[44], interval, start_date, end_date)
        TATACONSUM = get_data(ticker[45], interval, start_date, end_date)
        TATAMOTORS = get_data(ticker[46], interval, start_date, end_date)
        TATASTEEL = get_data(ticker[47], interval, start_date, end_date)
        TCS = get_data(ticker[48], interval, start_date, end_date)
        TECHM = get_data(ticker[49], interval, start_date, end_date)
        TITAN = get_data(ticker[50], interval, start_date, end_date)
        ULTRACEMCO = get_data(ticker[51], interval, start_date, end_date)
        WIPRO = get_data(ticker[52], interval, start_date, end_date)
        
        INDIAVIX=get_data(ticker_index[0], interval, start_date, end_date)
        CNXAUTO=get_data(ticker_index[1], interval, start_date, end_date)
        CNXPSUBANK=get_data(ticker_index[2], interval, start_date, end_date)
        CNXFMCG=get_data(ticker_index[4], interval, start_date, end_date)
        CNXMETAL=get_data(ticker_index[5], interval, start_date, end_date)
        CNXPSE=get_data(ticker_index[6], interval, start_date, end_date)
        CNXENERGY=get_data(ticker_index[7], interval, start_date, end_date)
        CNXPHARMA=get_data(ticker_index[8], interval, start_date, end_date)
        CNXMNC=get_data(ticker_index[9], interval, start_date, end_date)
        CNXIT=get_data(ticker_index[11], interval, start_date, end_date)
        CNXINFRA=get_data(ticker_index[12], interval, start_date, end_date)
        CNXCONSUM=get_data(ticker_index[13], interval, start_date, end_date)
        CNXSERVICE=get_data(ticker_index[14], interval, start_date, end_date)
        CNXCMDT=get_data(ticker_index[15], interval, start_date, end_date)
        CNXREALTY=get_data(ticker_index[16], interval, start_date, end_date)
        CNXMEDIA=get_data(ticker_index[17], interval, start_date, end_date)
        CNXFIN=get_data(ticker_index[18], interval, start_date, end_date)     
        
              
        def cal_data(data):
        
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
        
        NSEI_maxima = cal_data(NSEI)
        NSEBANK_maxima = cal_data(NSEBANK)
        ADANIENT_maxima = cal_data(ADANIENT)
        ADANIPORTS_maxima = cal_data(ADANIPORTS)
        APOLLOHOSP_maxima = cal_data(APOLLOHOSP)
        ASIANPAINT_maxima = cal_data(ASIANPAINT)
        AXISBANK_maxima = cal_data(AXISBANK)
        BAJAJ_AUTO_maxima = cal_data(BAJAJ_AUTO)
        BAJAJFINSV_maxima = cal_data(BAJAJFINSV)
        BAJFINANCE_maxima = cal_data(BAJFINANCE)
        BHARTIARTL_maxima = cal_data(BHARTIARTL)
        BPCL_maxima = cal_data(BPCL)
        BRITANNIA_maxima = cal_data(BRITANNIA)
        CIPLA_maxima = cal_data(CIPLA)
        COALINDIA_maxima = cal_data(COALINDIA)
        DIVISLAB_maxima = cal_data(DIVISLAB)
        DRREDDY_maxima = cal_data(DRREDDY)
        EICHERMOT_maxima = cal_data(EICHERMOT)
        GRASIM_maxima = cal_data(GRASIM)
        HCLTECH_maxima = cal_data(HCLTECH)
        HDFCBANK_maxima = cal_data(HDFCBANK)
        HDFCLIFE_maxima = cal_data(HDFCLIFE)
        HEROMOTOCO_maxima = cal_data(HEROMOTOCO)
        HINDALCO_maxima = cal_data(HINDALCO)
        HINDUNILVR_maxima = cal_data(HINDUNILVR)
        ICICIBANK_maxima = cal_data(ICICIBANK)
        INDUSINDBK_maxima = cal_data(INDUSINDBK)
        INFY_maxima = cal_data(INFY)
        ITC_maxima = cal_data(ITC)
        JSWSTEEL_maxima = cal_data(JSWSTEEL)
        KOTAKBANK_maxima = cal_data(KOTAKBANK)
        LT_maxima = cal_data(LT)
        LTIM_maxima = cal_data(LTIM)
        M_M_maxima = cal_data(M_M)
        MARUTI_maxima = cal_data(MARUTI)
        NESTLEIND_maxima = cal_data(NESTLEIND)
        NTPC_maxima = cal_data(NTPC)
        ONGC_maxima = cal_data(ONGC)
        POWERGRID_maxima = cal_data(POWERGRID)
        RELIANCE_maxima = cal_data(RELIANCE)
        SBILIFE_maxima = cal_data(SBILIFE)
        SBIN_maxima = cal_data(SBIN)
        SHRIRAMFIN_maxima = cal_data(SHRIRAMFIN)
        SUNPHARMA_maxima = cal_data(SUNPHARMA)
        TATACONSUM_maxima = cal_data(TATACONSUM)
        TATAMOTORS_maxima = cal_data(TATAMOTORS)
        TATASTEEL_maxima = cal_data(TATASTEEL)
        TCS_maxima = cal_data(TCS)
        TECHM_maxima = cal_data(TECHM)
        TITAN_maxima = cal_data(TITAN)
        ULTRACEMCO_maxima = cal_data(ULTRACEMCO)
        WIPRO_maxima = cal_data(WIPRO)
        
        # INDIAVIX_maxima = cal_data(INDIAVIX)
        CNXAUTO_maxima = cal_data(CNXAUTO)
        CNXPSUBANK_maxima = cal_data(CNXPSUBANK)
        CNXFMCG_maxima = cal_data(CNXFMCG)
        CNXMETAL_maxima = cal_data(CNXMETAL)
        CNXPSE_maxima = cal_data(CNXPSE)
        CNXENERGY_maxima = cal_data(CNXENERGY)
        # CNXPHARMA_maxima = cal_data(CNXPHARMA)
        CNXMNC_maxima = cal_data(CNXMNC)
        CNXIT_maxima = cal_data(CNXIT)
        CNXINFRA_maxima = cal_data(CNXINFRA)
        CNXCONSUM_maxima = cal_data(CNXCONSUM)
        CNXSERVICE_maxima = cal_data(CNXSERVICE)
        CNXCMDT_maxima = cal_data(CNXCMDT)
        CNXREALTY_maxima = cal_data(CNXREALTY)
        CNXMEDIA_maxima = cal_data(CNXMEDIA)
        # CNXFIN_maxima = cal_data(CNXFIN)
        
        dataframes = [
                            NSEI_maxima,NSEBANK_maxima,ADANIENT_maxima,ADANIPORTS_maxima,APOLLOHOSP_maxima,ASIANPAINT_maxima,AXISBANK_maxima,BAJAJ_AUTO_maxima,
                            BAJAJFINSV_maxima,BAJFINANCE_maxima,BHARTIARTL_maxima,BPCL_maxima,BRITANNIA_maxima,CIPLA_maxima,COALINDIA_maxima,DIVISLAB_maxima,
                            DRREDDY_maxima,EICHERMOT_maxima,GRASIM_maxima,HCLTECH_maxima,HDFCBANK_maxima,HDFCLIFE_maxima,HEROMOTOCO_maxima,HINDALCO_maxima,
                            HINDUNILVR_maxima,ICICIBANK_maxima,INDUSINDBK_maxima,INFY_maxima,ITC_maxima,JSWSTEEL_maxima,KOTAKBANK_maxima,LT_maxima,LTIM_maxima,
                            M_M_maxima,MARUTI_maxima,NESTLEIND_maxima,NTPC_maxima,ONGC_maxima,POWERGRID_maxima,RELIANCE_maxima,SBILIFE_maxima,SBIN_maxima,
                            SUNPHARMA_maxima,TATACONSUM_maxima,TATAMOTORS_maxima,TATASTEEL_maxima,TCS_maxima,TECHM_maxima,TITAN_maxima,ULTRACEMCO_maxima,WIPRO_maxima]
        dataframes_weighted = [
                            NSEI_maxima,NSEBANK_maxima,AXISBANK_maxima,BHARTIARTL_maxima ,HDFCBANK_maxima,ICICIBANK_maxima,
                            INFY_maxima,ITC_maxima,RELIANCE_maxima,SBIN_maxima,TCS_maxima,LT_maxima]
        dataframes_index =  [
                            CNXAUTO_maxima,CNXPSUBANK_maxima,CNXFMCG_maxima,CNXMETAL_maxima,CNXPSE_maxima,
                            CNXENERGY_maxima,CNXMNC_maxima,CNXIT_maxima,CNXINFRA_maxima,CNXCONSUM_maxima, 
                            CNXSERVICE_maxima,CNXCMDT_maxima,CNXREALTY_maxima,CNXMEDIA_maxima]#,CNXFIN_maxima]CNXPHARMA_maxima
        
        dataframes_index_weighted =  [CNXAUTO_maxima,CNXFMCG_maxima,CNXENERGY_maxima,CNXIT_maxima]#,CNXFIN_maxima]
            
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
        
        merged_df = calculate_active_occurrences(dataframes)      
        merged_df['close']=NSEI['close']
        Analysis_data = merged_df[['close', 'Short', 'Long']]
        Analysis_data_df = Analysis_data.sort_index(ascending=False)
                    
        merged_df_weighted = calculate_active_occurrences(dataframes_weighted) 
        merged_df_weighted['close'] = NSEI['close']
        Analysis_data_weighted = merged_df_weighted[['close', 'Short', 'Long']]
        Analysis_data_weighted = Analysis_data_weighted.sort_index(ascending=False)
        
        merged_df_index = calculate_active_occurrences(dataframes_index) 
        merged_df_index['close']=NSEI['close']
        Analysis_data_index = merged_df_index[['close', 'Short', 'Long']]
        Analysis_data_index = Analysis_data_index.sort_index(ascending=False)  
        
        merged_df_weighted_index = calculate_active_occurrences(dataframes_index_weighted) 
        merged_df_weighted_index['close']=NSEI['close']
        Analysis_data_index_weighted = merged_df_weighted_index[['close', 'Short', 'Long']]
        Analysis_data_index_weighted = Analysis_data_index_weighted.sort_index(ascending=False) 
        
        Analysis_data_weighted = Analysis_data_weighted.rename(columns={'Long': 'Long_W','Short': 'Short_W',})
        Analysis_data_weighted.drop(columns=['close'], inplace=True)
        Analysis_data_index = Analysis_data_index.rename(columns={'Long': 'Long_I','Short': 'Short_I',})  
        Analysis_data_index.drop(columns=['close'], inplace=True)
        
        Analysis_data_index_weighted = Analysis_data_index_weighted.rename(columns={'Long': 'Long_I_W','Short': 'Short_I_W',})  
        Analysis_data_index_weighted.drop(columns=['close'], inplace=True)
        
        combined_df = pd.concat([Analysis_data,Analysis_data_weighted, Analysis_data_index,Analysis_data_index_weighted], axis=1)  
        combined_df= combined_df.sort_index(ascending=False)
        # combined_df = combined_df[['close','Short_I_W', 'Short_I', 'Short','Long_I_W','Long_I','Long']] 
        
        def apply_color_based_on_previous_value(val):
            if val > apply_color_based_on_previous_value.previous_value:
                apply_color_based_on_previous_value.previous_value = val
                return 'background-color: green'
            else:
                apply_color_based_on_previous_value.previous_value = val
                return 'background-color: red'    
        apply_color_based_on_previous_value.previous_value = combined_df['close'].iloc[0]      
              
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
        
        styled_df = combined_df.style.applymap(apply_color_long_i, subset=['Long_I']) \
                                      .applymap(apply_color_short_i, subset=['Short_I']) \
                                      .applymap(apply_color_long, subset=['Long']) \
                                      .applymap(apply_color_short, subset=['Short']) \
                                      .applymap(apply_color_long_w, subset=['Long_W']) \
                                      .applymap(apply_color_short_w, subset=['Short_W']) \
                                      .applymap(apply_color_long_w_i, subset=['Long_I_W']) \
                                      .applymap(apply_color_short_w_i, subset=['Short_I_W']) \
                                      .apply(lambda x: x.apply(apply_color_based_on_previous_value), subset=['close']) 
                                      
        st.write(f"<span style='color: red; font-size: 18px;'>General calculations : {len(dataframes)} : Indices calculations {len(dataframes_index)} : Weighted calculations: {len(dataframes_weighted)} :  Index Weighted calculations: {len(dataframes_index_weighted)}</span>", unsafe_allow_html=True)

        st.dataframe(data=styled_df, width=1200, height=600, use_container_width=False, hide_index=False, column_order=['close','Short_I_W', 'Short_I', 'Short','Long_I_W','Long_I','Long'], column_config=None)
                
        rb_bounds   = calc_rangebreak(NSEI.index.to_series())
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,shared_yaxes=True,vertical_spacing=0.05,row_heights=[0.3, 0.1])
        fig.add_trace(go.Candlestick(
                x=NSEI.index,
                open=NSEI['open'],
                high=NSEI['high'],
                low=NSEI['low'],
                close=NSEI['close'],
                name='NSEI'), row=1, col=1 )            
        fig.add_trace(go.Scatter(
                x=NSEI.index,
                y=Analysis_data['Short'], 
                mode='lines',
                line=dict(color="red",width=3),
                name='Lower_high'), row=2, col=1)      
        fig.add_trace(go.Scatter(
                x=NSEI.index,
                y=Analysis_data['Long'], 
                mode='lines',
                line=dict(color="green",width=3),
                name='Higher_high'), row=2, col=1)   
        fig.add_trace(go.Scatter(
                x=NSEI.index,
                y=[10] * len(NSEI), 
                mode='lines',
                line=dict(color="orange",width=3, dash='dash'),
                name='th_0'), row=2, col=1)
        fig.add_trace(go.Scatter(
                x=NSEI.index,
                y=[15] * len(NSEI), 
                mode='lines',
                line=dict(color="red",width=3, dash='dash'),
                name='th_1'), row=2, col=1)            
        fig.update_layout(
                title='Asset Data',
                xaxis_title='time',
                yaxis_title='Live_levels',
                template='seaborn' ,
                xaxis_rangeslider_visible=False,
                width=1200,height=1600)
        (fig.update_xaxes(rangebreaks=rb_bounds))
        st.plotly_chart(fig)
        st.set_option('deprecation.showPyplotGlobalUse', False)
    
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
