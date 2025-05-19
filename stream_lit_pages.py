# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:24:53 2024

@author: kuldeep.rana
"""
import streamlit  as st
import warnings
from st_pages import Page, show_pages, add_page_title
import streamlit_authenticator as stauth

from yaml.loader  import SafeLoader
from streamlit_authenticator.utilities.exceptions import (CredentialsError,ForgotError,LoginError,RegisterError,ResetError,UpdateError) 
from yaml.loader  import SafeLoader
import yaml


# warnings.filterwarnings('ignore')
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)
authenticator = stauth.Authenticate(config['credentials'],config['cookie']['name'], config['cookie']['key'],config['cookie']['expiry_days'],config['pre-authorized'])
authenticator.login()
st.write(f'Welcome to AI ML *{st.session_state["name"]}*')   


if st.session_state["authentication_status"]:
    add_page_title("Website for advance analysis")
    show_pages([# Page("stream_lit_maxima_main.py", "Maxima_minima", "🏠"),    
            Page("stream_lit_maxima_main.py", "Maxima_minima", "🏠"),
            Page("multi-tickers-yf.py", "Additiona_analysis","📊"),])
    authenticator.logout(location='main')



if st.session_state["authentication_status"] is False:
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
