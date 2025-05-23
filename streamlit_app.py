import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from streamlit_authenticator.utilities.exceptions import (LoginError, RegisterError, ForgotError, ResetError, UpdateError)

# Load configuration from YAML file
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Initialize authenticator with 30-minute cookie expiry
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    cookie_expiry_days=0.0208  # 30 minutes
)

# Streamlit page configuration
# st.set_page_config(page_title="Login Page", page_icon="🔒", layout="centered")

# Custom CSS for better styling
st.markdown("""
    <style>
        .main { 
            background-color: #f0f2f6;
            padding: 2rem;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .error {
            color: #ff4b4b;
            text-align: center;
        }
        .success {
            color: #4CAF50;
            text-align: center;
        }
        .sidebar .sidebar-content {
            background-color: #ffffff;
        }
    </style>
""", unsafe_allow_html=True)

# Login page layout
st.title("Welcome to the Financial Analysis Platform")
st.markdown("Please log in to access market analysis tools.")

# Login form
try:
    authenticator.login(fields={'Form name': 'Login', 'Username': 'Username', 'Password': 'Password', 'Login': 'Login'}, location='main')
except LoginError as e:
    st.error(f"Login failed: {e}")

# Handle authentication status
if st.session_state["authentication_status"]:
    st.markdown(f"<p class='success'>Welcome, {st.session_state['name']}!</p>", unsafe_allow_html=True)
    st.markdown("You have successfully logged in. Your session will remain active for 30 minutes.")
    authenticator.logout('Logout', 'main')
elif st.session_state["authentication_status"] is False:
    st.markdown("<p class='error'>Username or password is incorrect.</p>", unsafe_allow_html=True)
elif st.session_state["authentication_status"] is None:
    st.markdown("Please enter your credentials to log in.")

# Sidebar for user access actions
st.sidebar.header("User Management")

# Registration form
if st.sidebar.checkbox("Register New User"):
    try:
        email, username, name = authenticator.register_user(
            fields={'Form name': 'Register', 'Email': 'Email', 'Username': 'Username', 'Password': 'Password', 'Repeat password': 'Repeat Password', 'Register': 'Register'},
            pre_authorization=False,
            location='sidebar'
        )
        if email:
            st.sidebar.success("User registered successfully!")
            with open('config.yaml', 'w', encoding='utf-8') as file:
                yaml.dump(config, file, default_flow_style=False)
    except RegisterError as e:
        st.sidebar.error(f"Registration failed: {e}")

# Forgot password
if st.sidebar.checkbox("Forgot Password"):
    try:
        username, email, new_password = authenticator.forgot_password(
            fields={'Form name': 'Forgot Password', 'Username': 'Username', 'Email': 'Email', 'Submit': 'Submit'},
            location='sidebar'
        )
        if username:
            st.sidebar.success("New password sent securely.")
        else:
            st.sidebar.error("Username not found.")
    except ForgotError as e:
        st.sidebar.error(f"Error: {e}")

# Forgot username
if st.sidebar.checkbox("Forgot Username"):
    try:
        username, email = authenticator.forgot_username(
            fields={'Form name': 'Forgot Username', 'Email': 'Email', 'Submit': 'Submit'},
            location='sidebar'
        )
        if username:
            st.sidebar.success("Username sent securely.")
        else:
            st.sidebar.error("Email not found.")
    except ForgotError as e:
        st.sidebar.error(f"Error: {e}")

# Update username
if st.sidebar.checkbox("Update Username") and st.session_state["authentication_status"]:
    try:
        if authenticator.update_user_details(
            st.session_state["username"],
            fields={'Form name': 'Update Username', 'New username': 'New Username', 'Submit': 'Update'},
            location='sidebar'
        ):
            st.sidebar.success("Username updated successfully.")
            with open('config.yaml', 'w', encoding='utf-8') as file:
                yaml.dump(config, file, default_flow_style=False)
    except UpdateError as e:
        st.sidebar.error(f"Update failed: {e}")

# Reset password
if st.sidebar.checkbox("Reset Password") and st.session_state["authentication_status"]:
    try:
        if authenticator.reset_password(
            st.session_state["username"],
            fields={'Form name': 'Reset Password', 'Current password': 'Current Password', 'New password': 'New Password', 'Repeat new password': 'Repeat New Password', 'Submit': 'Reset'},
            location='sidebar'
        ):
            st.sidebar.success("Password reset successfully.")
            with open('config.yaml', 'w', encoding='utf-8') as file:
                yaml.dump(config, file, default_flow_style=False)
    except (ResetError, CredentialsError) as e:
        st.sidebar.error(f"Password reset failed: {e}")