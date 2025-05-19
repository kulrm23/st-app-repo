import streamlit as st
import numpy as np
from scipy.stats import norm

def calculate_option_price_and_greeks(S, K, T_days, r, sigma, option_type):
    """
    Calculate the theoretical option price and Greeks (Delta, Gamma, Theta, Vega, Rho) for a European option.
    
    Parameters:
        S (float): Current stock price
        K (float): Strike price of the option
        T_days (float): Time to expiration (in days)
        r (float): Risk-free interest rate
        sigma (float): Implied volatility
        option_type (str): Type of option ('call' or 'put')
    
    Returns:
        dict: Dictionary containing calculated option price and Greeks
    """
    T_years = T_days / 365.0  # Convert time to expiration from days to years
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T_years) / (sigma * np.sqrt(T_years))
    d2 = d1 - sigma * np.sqrt(T_years)
    
    # Calculate option price
    if option_type == 'call':
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T_years) * norm.cdf(d2)
    elif option_type == 'put':
        option_price = K * np.exp(-r * T_years) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Please specify 'call' or 'put'.")
    
    # Calculate Greeks
    Delta = norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1
    Gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T_years))
    Theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T_years)) / 365.0 - r * K * np.exp(-r * T_years) * norm.cdf(d2) / 365.0
    Vega = S * np.exp(-d1**2 / 2) * np.sqrt(T_years) / 100.0  # Vega in percentage points
    Rho = K * T_years * np.exp(-r * T_years) * norm.cdf(d2) if option_type == 'call' else -K * T_years * np.exp(-r * T_years) * norm.cdf(-d2)
    
    return {
        "Option Price": option_price,
        "Delta": Delta,
        "Gamma": Gamma,
        "Theta": Theta,
        "Vega": Vega,
        "Rho": Rho
    }

def main():
    # Set page title and layout
    st.set_page_config(page_title="Options Pricing and Greeks Calculator", layout="wide")
    
    # Title and description with calculator icon
    st.title("📊 Options Pricing and Greeks Calculator")
    st.write("""
        This app calculates the theoretical price and Greeks (Delta, Gamma, Theta, Vega, Rho) for a European option.
        Enter the option parameters and select the option type (call or put) below.
    """)
    
    # Input form for option parameters
    st.header("Option Parameters")
    S = st.number_input("Current Stock Price (S)", min_value=0.01, step=0.01)
    K = st.number_input("Strike Price (K)", min_value=0.01, step=0.01)
    T_days = st.number_input("Time to Expiration (T) [days]", min_value=1, step=1)
    r = st.number_input("Risk-Free Interest Rate (r) [%]", min_value=0.0, step=0.01) / 100.0
    sigma = st.number_input("Implied Volatility (σ) [%]", min_value=0.0, step=0.01) / 100.0
    option_type = st.selectbox("Option Type", options=['call', 'put'])
    
    # Calculate option price and Greeks on button click
    if st.button("Calculate"):
        results = calculate_option_price_and_greeks(S, K, T_days, r, sigma, option_type)
        st.header("Option Pricing and Greeks")
        st.write(f"Theoretical {option_type.capitalize()} Option Price: {results['Option Price']:.4f}")
        st.write(f"Delta (Δ): {results['Delta']:.4f}")
        st.write(f"Gamma (Γ): {results['Gamma']:.4f}")
        st.write(f"Theta (Θ): {results['Theta']:.4f}")
        st.write(f"Vega (V): {results['Vega']:.4f}")
        st.write(f"Rho (ρ): {results['Rho']:.4f}")

if __name__ == "__main__":
    main()
