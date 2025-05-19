# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 08:57:24 2024

@author: kuldeep.rana
"""

import cmath
import math
import time
import yfinance  as yf
import warnings
import os
import pandas            as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import streamlit            as st
from plotly.subplots import make_subplots    
from datetime import datetime, timedelta
from yaml.loader     import SafeLoader
import streamlit_authenticator as stauth
import yaml


from streamlit_authenticator.utilities.exceptions import (CredentialsError,ForgotError,LoginError,RegisterError,ResetError,UpdateError) 

warnings.filterwarnings('ignore')
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(config['credentials'],config['cookie']['name'], config['cookie']['key'],config['cookie']['expiry_days'],config['pre-authorized'])
authenticator.login()

if st.session_state["authentication_status"]:
    authenticator.logout(location='sidebar')
    st.write(f'Welcome to AI ML *{st.session_state["name"]}*') 


    t     = 0    
    def get_data(ticker, interval, start, end):
        yfObj = yf.Ticker(ticker)
        data = yfObj.history(start=start, end=end, interval=interval)
        return data
    def get_date_range():
        today = datetime.today()
        start_dates = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(61)]  # Past 2 months
        end_dates = [(today + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(4)]  # Next 2 days
        return start_dates, end_dates
    def Zroot(L):
        return 0.5 * L * 1j  
    def Zjoint(L, t):
        exponent = 3 * math.pi / 45 * math.sin(t + math.pi / 6 * L)
        term     = (-1) ** L * cmath.exp(exponent) * 1j 
        return Zroot(L) + term
    def Ztip(L, t):
        exponent = 17 * math.pi / 45 * math.sin(t + math.pi / 6 * L)
        term     = (-1) ** L * cmath.exp(exponent) * 1j  
        return Zroot(L) + term
    def fomat_convert(L,result_joint):
        data_joint   = complex(L - result_joint*100*1)
        real_joint   = round(data_joint.real, 2)
        z_simplified = float(real_joint)    
        return z_simplified
    def calc_rangebreak(time_series:pd.Series):           
        timedeltas = time_series.diff()            
        if len(time_series) < 2:
            return []            
        missing_times = np.where([timedeltas > timedeltas.median()*1.5])[1]
        off = pd.Timedelta(seconds=0.0001)
        rb = [{'bounds': [str((time_series.iloc[t-1]+off)), str((time_series.iloc[t]-off))]} for t in missing_times]
        return rb
    def fractal_dimension_index(data, window):
        diff        = np.diff(data)
        sum_diff    = np.zeros(len(diff) - window + 1)
        sum_sq_diff = np.zeros(len(diff) - window + 1)        
        for i in range(window):
            sum_diff += np.abs(diff[i:len(diff) - window + 1 + i])
            sum_sq_diff += diff[i:len(diff) - window + 1 + i] ** 2        
        fdi = np.log(sum_sq_diff / sum_diff) / np.log(window)
        
        return fdi

    
    intervals = ['5m', '15m', '30m', '1h', '1d', '5d', '1wk', '1mo', '3mo','1m', '2m']
    interval = st.selectbox("Select Interval", intervals)
    start_dates, end_dates = get_date_range()
    start_date = st.date_input("Start Date", datetime.today() - timedelta(days=3))
    end_date = st.date_input("End Date", datetime.today() + timedelta(days=1))
    
    start_time = time.time()
    ticker = ['^NSEI', 'GC=F', '^NSEBANK', 'ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJAJFINSV.NS', 'BAJFINANCE.NS', 'BHARTIARTL.NS', 'BPCL.NS', 'BRITANNIA.NS', 'CIPLA.NS', 'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS',
              'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'INFY.NS', 'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'LTIM.NS', 'M&M.NS', 'MARUTI.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS', 'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS', 'SHRIRAMFIN.NS', 'SUNPHARMA.NS', 'TATACONSUM.NS', 'TATAMOTORS.NS', 'TATASTEEL.NS', 'TCS.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'WIPRO.NS']
    
    ticker = st.selectbox("Select Ticker", ticker)
    
    if st.button("Analyse"):
        
        df_orig = get_data(ticker, interval, start_date, end_date)
        df_Z_joint = pd.DataFrame(columns=['Z_joint'])
        df_Z_tip = pd.DataFrame(columns=['Z_tip'])
        
        L = float(df_orig['Open'].iloc[0])
        for _ in range(len(df_orig)):
            result_joint = Zjoint(L, t)  
            result_tip = Ztip(L, t) 
            z_joint = fomat_convert(L, result_joint) 
            Z_tip = fomat_convert(L, result_tip)  
        
            L += result_joint.real
            t += result_tip.imag
        
            df_Z_joint.loc[len(df_Z_joint)] = [z_joint]
            df_Z_tip.loc[len(df_Z_tip)] = [Z_tip]
    
        df_Z_joint_high  = max(df_Z_joint['Z_joint'])
        df_Z_joint_low   = min(df_Z_joint['Z_joint'])
        Z_joint_diff     = round(df_Z_joint_high-df_Z_joint_low)
           
        df_Z_joint_high1 = df_Z_joint['Z_joint'].nlargest(4).iloc[-3]
        df_Z_joint_low1  = df_Z_joint['Z_joint'].nsmallest(4).iloc[3]
        Z_joint_diff1    = round(df_Z_joint_high1-df_Z_joint_low1)
        
        df_Z_tip_high    = max(df_Z_tip['Z_tip'])
        df_Z_tip_low     = min(df_Z_tip['Z_tip'])
        Z_tip_diff       = round(df_Z_tip_high-df_Z_tip_low)
        
        df_Z_tip_high1   = df_Z_tip['Z_tip'].nlargest(2).iloc[-1]
        df_Z_tip_low1    = df_Z_tip['Z_tip'].nsmallest(2).iloc[1]
        Z_tip_diff1      = round(df_Z_tip_high1-df_Z_tip_low1)
    
        average_Z_joint  = round((df_Z_joint['Z_joint']).mean())
        average_Z_tip    = round((df_Z_tip['Z_tip']).mean())
        
        mean_high        = (df_Z_tip_high + df_Z_joint_high)/2
        mean_low         = (df_Z_tip_low  + df_Z_joint_low)/2
            
        # df_Z_joint['df_Z_joint_high'], df_Z_joint['df_Z_joint_low'], df_Z_tip['df_Z_tip_high'], df_Z_tip['df_Z_tip_low'], df_Z_joint['df_Z_joint_high1'], df_Z_joint['df_Z_joint_low1'], df_Z_tip['df_Z_tip_high1'], df_Z_tip['df_Z_tip_low1'], df_Z_joint['average_Z_joint'], df_Z_joint['mean_high'] = df_Z_joint_high, df_Z_joint_low, df_Z_tip_high, df_Z_tip_low, df_Z_joint_high1, df_Z_joint_low1, df_Z_tip_high1, df_Z_tip_low1, average_Z_joint, mean_high
        # plt.plot(df_orig['Close'], label='close', color='red', linewidth=3)
        # plt.plot(df_Z_joint['df_Z_joint_high'], label='3. Joint_High', color='blue', linewidth=3)
        # plt.plot(df_Z_joint['df_Z_joint_low'], label='4. Joint_Low', color='blue', linewidth=3)
        # plt.plot(df_Z_tip['df_Z_tip_high'], label='5. Tip_High', color='red', linewidth=3)
        # plt.plot(df_Z_tip['df_Z_tip_low'], label='6. Tip_Low', color='red', linewidth=3)
        # plt.plot(df_Z_joint['mean_high'], label='7. Mean_High', color='black')
        # plt.plot(df_Z_joint['average_Z_joint'], label='8. Avg_Z_Joint', color='black')
        # plt.fill_between(df_Z_joint.index, df_Z_joint['df_Z_joint_high'], df_Z_tip['df_Z_tip_high'], color='red', alpha=0.2)
        # plt.fill_between(df_Z_joint.index, df_Z_joint['df_Z_joint_low'], df_Z_tip['df_Z_tip_low'], color='teal', alpha=0.2)
        # title_text = f'Test-----{L}\nJoint----{Z_joint_diff}\nTip------{Z_tip_diff}\nDate----- to '
        # title = plt.title(title_text, loc='left')
        # title.set_bbox({'color': 'lightgrey', 'alpha': 0.5})
        # title.set_color('blue')
        # plt.legend(loc='upper left')
        # plt.grid()
        # st.pyplot()  
    
        df_Z_joint_high = [df_Z_joint_high] * len(df_orig.index)
        df_Z_joint_low = [df_Z_joint_low] * len(df_orig.index)
        df_Z_tip_high = [df_Z_tip_high] * len(df_orig.index)
        df_Z_tip_low = [df_Z_tip_low] * len(df_orig.index)
        mean_high = [mean_high] * len(df_orig.index)
        average_Z_joint = [average_Z_joint] * len(df_orig.index)
        
        rb_bounds   = calc_rangebreak(df_orig.index.to_series())
        
        price_data = df_orig['Close']
        fdi_values = fractal_dimension_index(price_data, window=3)
        max_fdi = np.max(fdi_values)
        min_fdi = np.min(fdi_values)
        max_fdi_array = np.full_like(fdi_values, max_fdi)
        min_fdi_array = np.full_like(fdi_values, min_fdi)
        fdi_values = pd.DataFrame({'fdi_values': fdi_values, 'max_fdi': max_fdi_array, 'min_fdi': min_fdi_array})
        
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,shared_yaxes=True,vertical_spacing=0.05,row_heights=[0.4,0.2])    
        fig.add_trace(go.Candlestick(x=df_orig.index,open=df_orig['Open'],high=df_orig['Close'],low=df_orig['Low'],close=df_orig['Close'],name='Close'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_orig.index, y=df_Z_joint_high, mode='markers', name='3. Joint_High', line=dict(color='blue', width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_orig.index, y=df_Z_joint_low, mode='markers', name='4. Joint_Low', line=dict(color='blue', width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_orig.index, y=df_Z_tip_high, mode='markers', name='5. Tip_High', line=dict(color='red', width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_orig.index, y=df_Z_tip_low, mode='markers', name='6. Tip_Low', line=dict(color='red', width=3)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_orig.index, y=mean_high, mode='markers', name='7. Mean_High', line=dict(color='yellow', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_orig.index, y=average_Z_joint, mode='markers', name='8. Avg_Z_Joint', line=dict(color='yellow', width=1)), row=1, col=1)
        
        fig.add_trace(go.Scatter(x=df_orig.index, y=fdi_values['fdi_values'], mode='lines', name='fdi_values', line=dict(color='red', width=1)),  row=2, col=1)
        fig.add_trace(go.Scatter(x=df_orig.index, y=fdi_values['max_fdi'   ], mode='markers', name='max_fdi', line=dict(color='red', width=1)),   row=2, col=1)
        fig.add_trace(go.Scatter(x=df_orig.index, y=fdi_values['min_fdi'   ], mode='markers', name='min_fdi', line=dict(color='green', width=1)), row=2, col=1)
        fig.update_yaxes(title_text='Fractal Dimension Index (FDI)', row=2, col=1)
        fig.update_layout(title='Asset Data',xaxis_title='time',yaxis_title='values',template='seaborn' ,xaxis_rangeslider_visible=False,width=1000,height=1200)
        (fig.update_xaxes(rangebreaks=rb_bounds))
        title_text = 'Test\nJoint\nTip\nDate'
    
    
    
        st.plotly_chart(fig)

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
