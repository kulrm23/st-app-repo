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

start_time = time.time() 

import streamlit_authenticator as stauth
import plotly.graph_objects    as go

from datetime        import datetime, timedelta
from plotly.subplots import make_subplots    
from scipy.signal    import argrelextrema
from collections     import deque
from yaml.loader     import SafeLoader
from json_reader     import read_config
from close_value     import CLV
from   database_setup           import database_setup

warnings.filterwarnings('ignore')

class maxima():
    def do_caluclations(self):
        global combined_df,sp1
        
        def refine_data(df):
            df = df.copy()
            df['time'] = df['time'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
            df['time'] = pd.to_datetime(df['time'])
            df['time'] = pd.to_datetime(df['time'] + timedelta(hours=5, minutes=30))
            return df
        def resample(df):
            df =df.copy()
            df.set_index('time', inplace=True)
            ohlc_columns = [col for col in df.columns if 'ltp' in col]
            agg_dict = {col: 'ohlc' for col in ohlc_columns}
            
            resampled_1min = df.resample('1T').agg(agg_dict)  
            resampled_1min.columns = ['_'.join(col) for col in resampled_1min.columns]
            resampled_1min.reset_index(inplace=True)
            return resampled_1min
        
            
        today_date = datetime.now().date()
        # today_date = "1/10/2024"
        today_date_start = "05/10/2024"
        data_test_db   = database_setup.read_last_hundred_maxima("shapi",today_date,today_date_start,150000) 
        data_test_db_ce = database_setup.read_last_hundred_maxima_ce("shapi",today_date,today_date_start,150000) 
        
        # data_test_db = data_test_db.dropna
        # data_test_db_ce = data_test_db_ce.dropna

        data_test_db = refine_data(data_test_db)
        data_test_db_ce = refine_data(data_test_db_ce)
        
        data_test_db  = resample(data_test_db)
        close_columns = [col for col in data_test_db.columns if col.endswith('_close')]
        data_test_db =  data_test_db[['time'] + close_columns]
        data_test_db.columns = [col.replace('_close', '') for col in data_test_db.columns]
        
        
        data_test_db_ce  = resample(data_test_db_ce)
        close_columns = [col for col in data_test_db_ce.columns if col.endswith('_close')]
        data_test_db_ce =  data_test_db_ce[['time'] + close_columns]
        data_test_db_ce.columns = [col.replace('_close', '') for col in data_test_db_ce.columns]     
        
        
        historical_fut = data_test_db.iloc[:, [12]].join(data_test_db[['time']])
        historical_fut.rename(columns={'fut1_ltp': 'Close'}, inplace=True)
        historical_fut['time_column'] = pd.to_datetime(historical_fut['time'])
        historical_fut.set_index('time_column', inplace=True)
        
        sp1 = data_test_db.iloc[:, [1]].join(data_test_db[['time']])
        sp2 = data_test_db.iloc[:, [2]].join(data_test_db[['time']])
        sp3 = data_test_db.iloc[:, [3]].join(data_test_db[['time']])
        sp4 = data_test_db.iloc[:, [4]].join(data_test_db[['time']])
        sp5 = data_test_db.iloc[:, [5]].join(data_test_db[['time']])
        sp6 = data_test_db.iloc[:, [6]].join(data_test_db[['time']])
        sp7 = data_test_db.iloc[:, [7]].join(data_test_db[['time']])
        sp8 = data_test_db.iloc[:, [8]].join(data_test_db[['time']])
        sp9 = data_test_db.iloc[:, [9]].join(data_test_db[['time']])
        
        sp1.rename(columns={'opti_pe_1_ltp': 'Close'}, inplace=True)
        sp2.rename(columns={'opti_pe_2_ltp': 'Close'}, inplace=True)
        sp3.rename(columns={'opti_pe_3_ltp': 'Close'}, inplace=True)
        sp4.rename(columns={'opti_pe_4_ltp': 'Close'}, inplace=True)
        sp5.rename(columns={'opti_pe_5_ltp': 'Close'}, inplace=True)
        sp6.rename(columns={'opti_pe_6_ltp': 'Close'}, inplace=True)
        sp7.rename(columns={'opti_pe_7_ltp': 'Close'}, inplace=True)
        sp8.rename(columns={'opti_pe_8_ltp': 'Close'}, inplace=True)
        sp9.rename(columns={'opti_pe_9_ltp': 'Close'}, inplace=True)
        
        sp1['time_column'] = pd.to_datetime(sp1['time'])
        sp1.set_index('time_column', inplace=True)
        
        sp2['time_column'] = pd.to_datetime(sp2['time'])
        sp2.set_index('time_column', inplace=True)
        
        sp3['time_column'] = pd.to_datetime(sp3['time'])
        sp3.set_index('time_column', inplace=True)
        
        sp4['time_column'] = pd.to_datetime(sp4['time'])
        sp4.set_index('time_column', inplace=True)
        
        sp5['time_column'] = pd.to_datetime(sp5['time'])
        sp5.set_index('time_column', inplace=True)
        
        sp6['time_column'] = pd.to_datetime(sp6['time'])
        sp6.set_index('time_column', inplace=True)
        
        sp7['time_column'] = pd.to_datetime(sp7['time'])
        sp7.set_index('time_column', inplace=True)
        
        sp8['time_column'] = pd.to_datetime(sp8['time'])
        sp8.set_index('time_column', inplace=True)
              
        sp9['time_column'] = pd.to_datetime(sp9['time'])
        sp9.set_index('time_column', inplace=True)
        
        
        sp1_ce = data_test_db_ce.iloc[:, [1]].join(data_test_db_ce[['time']])
        sp2_ce = data_test_db_ce.iloc[:, [2]].join(data_test_db_ce[['time']])
        sp3_ce = data_test_db_ce.iloc[:, [3]].join(data_test_db_ce[['time']])
        sp4_ce = data_test_db_ce.iloc[:, [4]].join(data_test_db_ce[['time']])
        sp5_ce = data_test_db_ce.iloc[:, [5]].join(data_test_db_ce[['time']])
        sp6_ce = data_test_db_ce.iloc[:, [6]].join(data_test_db_ce[['time']])
        sp7_ce = data_test_db_ce.iloc[:, [7]].join(data_test_db_ce[['time']])
        sp8_ce = data_test_db_ce.iloc[:, [8]].join(data_test_db_ce[['time']])
        sp9_ce = data_test_db_ce.iloc[:, [9]].join(data_test_db_ce[['time']])
        
        
        # sp1_ce.rename(columns={'close_sp1': 'Close'}, inplace=True)
        # sp2_ce.rename(columns={'close_sp2': 'Close'}, inplace=True)
        # sp3_ce.rename(columns={'close_sp3': 'Close'}, inplace=True)
        # sp4_ce.rename(columns={'close_sp4': 'Close'}, inplace=True)
        # sp5_ce.rename(columns={'close_sp5': 'Close'}, inplace=True)
        # sp6_ce.rename(columns={'close_sp6': 'Close'}, inplace=True)
        # sp7_ce.rename(columns={'close_sp7': 'Close'}, inplace=True)
        # sp8_ce.rename(columns={'close_sp8': 'Close'}, inplace=True)
        # sp9_ce.rename(columns={'close_sp9': 'Close'}, inplace=True)
        
        sp1_ce.rename(columns={'opti_ce_1_ltp': 'Close'}, inplace=True)
        sp2_ce.rename(columns={'opti_ce_2_ltp': 'Close'}, inplace=True)
        sp3_ce.rename(columns={'opti_ce_3_ltp': 'Close'}, inplace=True)
        sp4_ce.rename(columns={'opti_ce_4_ltp': 'Close'}, inplace=True)
        sp5_ce.rename(columns={'opti_ce_5_ltp': 'Close'}, inplace=True)
        sp6_ce.rename(columns={'opti_ce_6_ltp': 'Close'}, inplace=True)
        sp7_ce.rename(columns={'opti_ce_7_ltp': 'Close'}, inplace=True)
        sp8_ce.rename(columns={'opti_ce_8_ltp': 'Close'}, inplace=True)
        sp9_ce.rename(columns={'opti_ce_9_ltp': 'Close'}, inplace=True)
        
        
        sp1_ce['time_column'] = pd.to_datetime(sp1_ce['time'])
        sp1_ce.set_index('time_column', inplace=True)
        
        sp2_ce['time_column'] = pd.to_datetime(sp2_ce['time'])
        sp2_ce.set_index('time_column', inplace=True)
        
        sp3_ce['time_column'] = pd.to_datetime(sp3_ce['time'])
        sp3_ce.set_index('time_column', inplace=True)
        
        sp4_ce['time_column'] = pd.to_datetime(sp4_ce['time'])
        sp4_ce.set_index('time_column', inplace=True)
        
        sp5_ce['time_column'] = pd.to_datetime(sp5_ce['time'])
        sp5_ce.set_index('time_column', inplace=True)
        
        sp6_ce['time_column'] = pd.to_datetime(sp6_ce['time'])
        sp6_ce.set_index('time_column', inplace=True)
        
        sp7_ce['time_column'] = pd.to_datetime(sp7_ce['time'])
        sp7_ce.set_index('time_column', inplace=True)
        
        sp8_ce['time_column'] = pd.to_datetime(sp8_ce['time'])
        sp8_ce.set_index('time_column', inplace=True)
              
        sp9_ce['time_column'] = pd.to_datetime(sp9_ce['time'])
        sp9_ce.set_index('time_column', inplace=True)       
        
        def cal_data(data):
            start_time = time.time()
            
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
            # data_dates = data.time_sp1
            data['time_column'] = pd.to_datetime(data['time'])
            data.set_index('time_column', inplace=True)
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
        close_column = sp5["close"].values
        merged_df = calculate_active_occurrences(dataframes)
        merged_df["close"] =  close_column
        Analysis_data = merged_df[['close', 'Short', 'Long']]
        Analysis_data = Analysis_data.rename(columns={'close':'close_pe','Long': 'Long_pe','Short': 'Short_pe'}) 
        Analysis_data_df = Analysis_data.sort_index(ascending=False)
        
        close_column_ce = sp1_ce["close"].values
        merged_df_ce = calculate_active_occurrences(dataframes_ce)
        merged_df_ce["close"] =  close_column_ce
        Analysis_data_ce = merged_df_ce[['close','Short', 'Long']]
        Analysis_data_ce = Analysis_data_ce.rename(columns={'close':'close_ce','Long': 'Long_ce','Short': 'Short_ce'}) 
        Analysis_data_df_ce = Analysis_data_ce.sort_index(ascending=False)
        
        close_column_fut = historical_fut["close"].values
        merged_df_fut = calculate_active_occurrences(dataframes_fut)
        merged_df_fut["close"] =  close_column_fut
        Analysis_data_fut = merged_df_fut[['close','Short', 'Long']]
        Analysis_data_fut = Analysis_data_fut.rename(columns={'close':'close_fut','Long': 'Long_fut','Short': 'Short_fut'}) 
        Analysis_data_df_fut = Analysis_data_fut.sort_index(ascending=False)
        
        Analysis_data_df.reset_index(drop=True,inplace=True)
        Analysis_data_df_ce.reset_index(drop=True,inplace=True)
        Analysis_data_df_fut.reset_index(drop=True,inplace=True)
        
        combined_df = pd.concat([Analysis_data_df,Analysis_data_df_ce,Analysis_data_df_fut], axis=1)  
        combined_df = combined_df.sort_index(ascending=False)
        combined_df["time"] = data_test_db["time"]
        # combined_df.set_index('time', inplace = True)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(elapsed_time)
        # database_setup.write("A_maxima",combined_df)
        
        return combined_df,sp1
# nseobj = maxima()
# nseobj.do_caluclations()

while True:
    if __name__ == '__main__':
        nseobj = maxima()
        nseobj.do_caluclations()
    time.sleep(80)
    
