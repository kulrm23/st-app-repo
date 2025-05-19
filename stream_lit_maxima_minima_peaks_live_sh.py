# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 09:37:20 2024

@author: kuldeep.rana

"""
import warnings
import time
import pandas   as pd

from   database_setup           import database_setup
from   calculate_maximas_helper import *
from   misc.data_processing     import data_processing
from   json_reader              import read_config
from   datetime                 import datetime,timedelta

warnings.filterwarnings('ignore')

data_processing = data_processing()


def output(data,order=12, K=2):    
    joined_output,data = detect_peaks_function_sh(data,order=12, K=2)
    # zones_data_sh(data,config["scale"],joined_output,True)
    return data,joined_output

def resample(df):
    df =df.copy()
    df.set_index('Datetime', inplace=True)
    # For example: '1T' for 1 minute, '5T' for 5 minutes, '1H' for 1 hour, etc.
    resampled_1min  = df['Close'].resample('1T').ohlc()
    # resampled_5min  = df['Close'].resample('5T').ohlc()
    # resampled_1hour = df['Close'].resample('1H').ohlc()
    resampled_1min  = resampled_1min.dropna()
    # resampled_5min  = resampled_5min.dropna()
    # resampled_1hour = resampled_1hour.dropna()
    
    resampled_1min.columns   = [col.capitalize() for col in resampled_1min.columns]
    resampled_1min.reset_index(inplace=True)
    # resampled_5min.columns   = [col.capitalize() for col in resampled_5min.columns]
    # resampled_5min.reset_index(inplace=True)
    # resampled_1hour.columns   = [col.capitalize() for col in resampled_1hour.columns]
    # resampled_1hour.reset_index(inplace=True)
    
    # return resampled_1min,resampled_5min,resampled_1hour
    return resampled_1min

def refine_data(df):
    df = df.copy()
    df["Datetime"] = df["Datetime"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
    df.columns     = [col.capitalize() for col in df.columns]
    df["Datetime"] = pd.to_datetime(df['Datetime'])
    df["Datetime"] = pd.to_datetime(df['Datetime'] + timedelta(hours=5, minutes=30))
    return df

# In[]

config = read_config(config_file="E:\containers\shapi\explore/config.json", readjson_with_comments=True) 
today_date = datetime.now().date()
today_date_start = "4/1/2024"
# today_date ="4/1/2024"
level_op = config["level_opti"]
level_op_c = 8

string_pe = config["string_asset_opti"] + "_" + config["asset_pe"] + f'_{level_op}_'
data_pe   = database_setup.read_last_hundred_sh("shapi",config["asset_pe"],config["string_asset_opti"],config["level_opti"],today_date,today_date_start,10000) 
data_pe   = data_processing.opti_processing_maxima_sh(data_pe,string_pe)
data_pe   = refine_data(data_pe) 
data_pe_1m = resample(data_pe)
data_label_pe,joined_output_pe    = output(data_pe_1m,order=4, K=2)

string_ce = config["string_asset_opti"] + "_" + config["asset_ce"] + f'_{level_op_c}_'
data_ce   = database_setup.read_last_hundred_sh("shapi",config["asset_ce"],config["string_asset_opti"],level_op_c,today_date,today_date_start,10000)
data_ce   = data_processing.opti_processing_maxima_sh(data_ce,string_ce)
data_ce   = refine_data(data_ce) 
data_ce_1m = resample(data_ce)
data_label_ce,joined_output_ce    = output(data_ce_1m,order=4, K=2)

string_fut = "fut1_"
data_fut   = database_setup.read_last_hundred_sh_fut("shapi",string_fut,today_date,today_date_start,10000) 
data_fut   = data_processing.fut_processing_sh(data_fut,string_fut)
data_fut   = refine_data(data_fut)
data_fut_1m = resample(data_fut)
data_label_fut,joined_output_fut    = output(data_fut_1m,order=4, K=2)
 

string_cash = "nc_" + config["asset_cash"] + "_"
data_cash   = database_setup.read_last_hundred_sh_cash_test("shapi",config["asset_cash"],today_date,today_date_start,10000) 
data_cash   = data_processing.fut_processing_sh(data_cash,string_cash)
data_cash   = refine_data(data_cash) 

data_cash_1m = resample(data_cash)
data_label_cash,joined_output_cash    = output(data_cash_1m,order=5, K=2)

# In[]

today_date = datetime.now().date()
today_date_start = "4/1/2024"

data_test_db   = database_setup.read_last_hundred("shapi")
columns=data_test_db.columns
data_test_db   = database_setup.read_last_hundred_maxima("shapi",today_date,today_date_start,5000) 


end_time = time.time()
elapsed_time = end_time - start_time
print("Time elapsed:", elapsed_time, "seconds")
    
# In[] resampled 1,5,and 1 hr

# data_processing = data_processing()
# config = read_config(config_file="E:\containers\shapi\explore/config.json", readjson_with_comments=True) 
# today_date = datetime.now().date()
# today_date_start = "4/1/2024"
# # today_date ="4/1/2024"
# level_op = config["level_opti"]
# level_op_c = 8

# string_pe = config["string_asset_opti"] + "_" + config["asset_pe"] + f'_{level_op}_'
# data_pe   = database_setup.read_last_hundred_sh("shapi",config["asset_pe"],config["string_asset_opti"],config["level_opti"],today_date,today_date_start,10000) 
# data_pe   = data_processing.opti_processing_maxima_sh(data_pe,string_pe)
# data_pe   = refine_data(data_pe) 
# data_pe_1m,data_pe_5m,data_pe_1hr = resample(data_pe)
# data_label_pe,joined_output_pe    = output(data_pe_1m,order=4, K=2)

# string_ce = config["string_asset_opti"] + "_" + config["asset_ce"] + f'_{level_op_c}_'
# data_ce   = database_setup.read_last_hundred_sh("shapi",config["asset_ce"],config["string_asset_opti"],level_op_c,today_date,today_date_start,10000)
# data_ce   = data_processing.opti_processing_maxima_sh(data_ce,string_ce)
# data_ce   = refine_data(data_ce) 
# data_ce_1m,data_ce_5m,data_ce_1hr = resample(data_ce)
# data_label_ce,joined_output_ce    = output(data_ce_1m,order=4, K=2)

# string_fut = "fut1_"
# data_fut   = database_setup.read_last_hundred_sh_fut("shapi",string_fut,today_date,today_date_start,10000) 
# data_fut   = data_processing.fut_processing_sh(data_fut,string_fut)
# data_fut   = refine_data(data_fut)
# data_fut_1m,data_fut_5m,data_fut_1hr = resample(data_fut)
# data_label_fut,joined_output_fut    = output(data_fut_5m,order=4, K=2)
 

# string_cash = "nc_" + config["asset_cash"] + "_"
# data_cash   = database_setup.read_last_hundred_sh_cash_test("shapi",config["asset_cash"],today_date,today_date_start,10000) 
# data_cash   = data_processing.fut_processing_sh(data_cash,string_cash)
# data_cash   = refine_data(data_cash) 

# data_cash_1m,data_cash_5m,data_cash_1hr = resample(data_cash)
# data_label_cash,joined_output_cash    = output(data_cash_5m,order=5, K=2)


# end_time = time.time()
# elapsed_time = end_time - start_time
# print("Time elapsed:", elapsed_time, "seconds")

 # In[Post-processing-Data-Output]

# def post_processing(data_list):        
#     data            = [[x[1] for x in sublist] for sublist in data_list]
#     timestamps      = [[x[0] for x in sublist] for sublist in data_list]
#     data_flat       = [item for sublist in data for item in sublist]
#     timestamps_flat = [item for sublist in timestamps for item in sublist]
#     df = pd.DataFrame({'Timestamp': timestamps_flat, 'Value': data_flat})
#     df.rename(columns={'Timestamp': 'time','Value': 'value',}, inplace=True)
    
#     return df

# data_list_hh   = out_hh 
# data_list_ll   = out_ll 
# data_list_hl   = out_hl
# data_list_lh   = out_lh
# df_hh          = post_processing(data_list_hh)
# df_ll          = post_processing(data_list_ll)
# df_lh          = post_processing(data_list_hl)
# df_hl          = post_processing(data_list_lh)

# joined_output1 = post_processing(joined_output_fut)

 # In[] Date-Write-db
 
# data.reset_index(inplace=True)
# data.columns = data.columns.astype(str)
# data.columns = data.columns.str.capitalize()

# database_setup.write_rhel('out_hh', df_hh)
# database_setup.write_rhel('out_hl', df_ll)
# database_setup.write_rhel('out_ll', df_lh)
# database_setup.write_rhel('out_lh', df_hl)

# database_setup.write('maxima_main', joined_output1) 