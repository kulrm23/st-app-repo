# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 11:36:35 2024

@author: kuldeep.rana
"""

import streamlit as st

def main():
    st.set_page_config(page_title="Blogging the Daily Analytics", layout="wide")    
    st.title("Welcome to the AI/Data-Science World Blogs")
    st.write(""" """)
    
    st.header("Blogs")
    st.header(""" Blog-1 """)
    st.image("long_indication.jpg", caption="Long_indication_maxima_index_29042024", use_column_width=True)
    st.markdown(""" Market reversed from low zone and at the same time Index maxima suggested strong up move as its value was 5
                The same move continued as its also confimed by general maxima calculations.\n
                
                1. indication: Maximas of indexes
                2. Low Zone confimation
                3. Near all time High
                4. Elections Sentiment.
                5. Market followed price action, 22600 ce marked high 160 from low of 80 
                
                \n
                Trade output- Exited at 110 due to busy in work made 2200 today with capital of 10k""")
                
    st.image("long_indication_chart_26_04_2024.jpg", caption="Long_indication_maxima_Nifty_29042024", use_column_width=True)
    st.markdown(""" \n ======================================================================================= \n
                    ===========================End of blog 1 ======================================  \n
                    =======================================================================================  \n""")

    # st.image("stock_market_image.jpg", caption="Stock Market Analysis", use_column_width=True)
      
    st.sidebar.markdown("---")
    # st.sidebar.write("Built with Streamlit")

if __name__ == "__main__":
    main()
