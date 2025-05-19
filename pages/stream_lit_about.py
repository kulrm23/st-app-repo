# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:36:35 2024

@author: kuldeep.rana
"""

import streamlit as st

def main():
    st.set_page_config(page_title="About Auther", layout="wide")    
    st.title("Welcome to the AI/Data-Science World")
    st.write(""" """)
    
    st.header("Meet the Author: Kuldeep rana")
    st.markdown(""" 
About Me \n
Welcome to AI-Analytics! I'm Kuldeep rana, a passionate data science engineer specializing in financial analytics and quantitative trading. With a background in both data science and finance, I am dedicated to leveraging technology and data-driven insights to navigate the complexities of the stock market.

My Journey \n
My journey into the world of data science and finance began during my academic pursuits, where I developed a keen interest in applying mathematical models and statistical techniques to analyze financial markets. Over the years, I have gained valuable experience working with leading financial institutions and technology companies, honing my skills in algorithmic trading, risk management, and quantitative analysis.

Expertise \n
As a data science engineer, I specialize in: \n

Financial Data Analysis: Extracting meaningful insights from market data using advanced statistical methods and machine learning algorithms.
Algorithmic Trading: Designing and implementing algorithmic trading strategies to optimize portfolio performance and mitigate risk.
Quantitative Modeling: Building mathematical models to forecast market trends, estimate asset prices, and assess investment opportunities.
Technology Integration: Leveraging cutting-edge technologies to develop robust and scalable solutions for real-time market analysis and automated trading.
Mission
My mission is to empower investors, traders, and finance enthusiasts with the tools and knowledge needed to make informed decisions in the ever-changing landscape of finance. Through AI-Analytics, I aim to share my expertise, insights, and resources to help individuals navigate the complexities of the stock market and achieve their financial goals.

What to Expect \n
On this platform, you can expect: \n

Insightful Articles: Engaging content covering a wide range of topics in finance, data science, and investment strategies.
Advanced Tools and Resources: Access to powerful analytical tools, models, and educational resources designed to enhance your understanding of the stock market.
Community Interaction: Join a vibrant community of like-minded individuals to share ideas, discuss trends, and collaborate on innovative projects.


Connect With Me \n
I'm excited to connect with fellow enthusiasts, investors, and professionals! Feel free to reach out, share your insights, or collaborate on exciting projects. You can contact me via:

Email: kuldeep7322@gmail.com

Let's explore the possibilities of data-driven finance together.

Let's Dive In! \n
Join me on this journey as we explore the fascinating intersection of data science and finance. Whether you're a seasoned investor or a curious learner, together we can uncover valuable insights and opportunities in the world of stock market analysis.""")
    
    # st.image("stock_market_image.jpg", caption="Stock Market Analysis", use_column_width=True)
      
    st.sidebar.markdown("---")
    # st.sidebar.write("Built with Streamlit")

if __name__ == "__main__":
    main()
