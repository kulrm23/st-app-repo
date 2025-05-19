# -*- coding: utf-8 -*-
"""
Created on Thu May 16 09:50:01 2024

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

def get_api_data_index_opti_v2(strike_price,right,interval,product_type,exchange_code,stock_code,expiry_date,session_token,from_date,to_date):        
    breeze                    = BreezeConnect(api_key="W57^039330`163143`w385Ug8404ORL1")
    breeze.generate_session(api_secret="y45173e22303284_=y`5IG9444@64S57", session_token=session_token)
    historical_data = [] 
    data_1m         = []     

    df_his = breeze.get_historical_data_v2(
                                        interval=interval,from_date=str(from_date), 
                                        to_date=str(to_date),stock_code=stock_code,
                                        exchange_code=exchange_code,product_type=product_type,
                                        expiry_date=str(expiry_date),right=right, strike_price=str(strike_price))
    if 'Success' in df_his:
        df_his = pd.DataFrame(df_his['Success'])
    return df_his

exchange = 'NF'
start_date       = "2024-05-14"
end_date         =  "2024-05-16"
expiry_date     = "2024-05-16"
expiry_date_fut = "2024-05-30"
ticker = 'NIFTY'
config_icic = read_config(config_file="config.json", readjson_with_comments=True) 
strike_price=22300
start_time = time.time()    
# sp1 = get_api_data_index_opti_v2(strike_price-200,config_icic["right"],config_icic["interval"],config_icic["product_type"],config_icic["exchange_code"],config_icic["stock_code"],str(expiry_date)+"T07:00:00.000Z",config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       
# sp2 = get_api_data_index_opti_v2(strike_price-150,config_icic["right"],config_icic["interval"],config_icic["product_type"],config_icic["exchange_code"],config_icic["stock_code"],str(expiry_date)+"T07:00:00.000Z",config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       
# sp3 = get_api_data_index_opti_v2(strike_price-100,config_icic["right"],config_icic["interval"],config_icic["product_type"],config_icic["exchange_code"],config_icic["stock_code"],str(expiry_date)+"T07:00:00.000Z",config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       
# sp4 = get_api_data_index_opti_v2(strike_price-50,config_icic["right"],config_icic["interval"],config_icic["product_type"],config_icic["exchange_code"],config_icic["stock_code"],str(expiry_date)+"T07:00:00.000Z",config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       
# sp5 = get_api_data_index_opti_v2(strike_price+50,config_icic["right"],config_icic["interval"],config_icic["product_type"],config_icic["exchange_code"],config_icic["stock_code"],str(expiry_date)+"T07:00:00.000Z",config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       
# sp6 = get_api_data_index_opti_v2(strike_price+100,config_icic["right"],config_icic["interval"],config_icic["product_type"],config_icic["exchange_code"],config_icic["stock_code"],str(expiry_date)+"T07:00:00.000Z",config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       
# sp7 = get_api_data_index_opti_v2(strike_price+150,config_icic["right"],config_icic["interval"],config_icic["product_type"],config_icic["exchange_code"],config_icic["stock_code"],str(expiry_date)+"T07:00:00.000Z",config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       
# sp8 = get_api_data_index_opti_v2(strike_price+200,config_icic["right"],config_icic["interval"],config_icic["product_type"],config_icic["exchange_code"],config_icic["stock_code"],str(expiry_date)+"T07:00:00.000Z",config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       
# sp9 = get_api_data_index_opti_v2(strike_price+250,config_icic["right"],config_icic["interval"],config_icic["product_type"],config_icic["exchange_code"],config_icic["stock_code"],str(expiry_date)+"T07:00:00.000Z",config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       
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
historical_fut = get_api_data_index_fut_v2(config_icic["right"],config_icic["interval_5"],config_icic["product_type_fut"],config_icic["exchange_code"],config_icic["stock_code"],expiry_date_fut,config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")



# df_opti_ce = get_api_data_index_opti_v2(current_strike_price,config_icic["right_c"],config_icic["interval"],config_icic["product_type"],config_icic["exchange_code"],config_icic["stock_code"],str(expiry_date)+"T07:00:00.000Z",config_icic["session_token"],str(start_date)+"T07:00:00.000Z",str(end_date)+"T07:00:00.000Z")       






