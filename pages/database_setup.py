# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 14:41:29 2023
@author: kuldeep.rana
"""

from   sqlalchemy                      import create_engine
from   get_ip_address                  import network
import pandas as pd

""" define the connection parameters"""

host                    = '192.168.137.15'
host2                   = '172.20.46.165'     
port                    = '5432'
database                = 'testdb_dev_1'
database1               = 'testdb_dev_1_01_02_2023'
database3               = 'prod_maxima'
user                    = 'postgres'
password                = 'admin'
conn_string             =  f'postgresql://{user}:{password}@{host}:{port}/{database}'   
conn_string1            =  f'postgresql://{user}:{password}@{host}:{port}/{database1}'   
conn_string2            =  f'postgresql://{user}:{password}@{host2}:{port}/{database3}'  
        
hostname_to_ping = "ubuntu-vm"
ip_class_import  = network()
host_ip          = network.get_ip_address(str(hostname_to_ping))

class database_setup():
    
    def write(table_name,df): 
        df=df
        engine = create_engine(conn_string)
        df.to_sql(table_name, engine, if_exists='append', index=False)
        # engine.close()
        
    def write_rhel(table_name,df): 
        df=df
        engine = create_engine(conn_string2)
        df.to_sql(table_name, engine, if_exists='append', index=False)
        # engine.close()
        
    def write_parms(table_name,df): 
        df=df
        engine = create_engine(conn_string)
        df.to_sql(table_name, engine, if_exists='append', index=False)
        # engine.close()    
        
    def write_o(table_name,df): 
        df=df
        engine = create_engine(conn_string)
        df.to_sql(table_name, engine,if_exists='replace', index=False) 
        # engine.close()
        
    def write_o_i(table_name,df): 
        df=df
        engine = create_engine(conn_string)
        df.to_sql(table_name, engine,if_exists='replace', index=True) 
        # engine.close()

    def read(table_name):  
        engine = create_engine(conn_string)
        # engine.close()   
        return  pd.read_sql_table(table_name, con=engine)
    def read1(table_name):  
        engine1 = create_engine(conn_string1)
        # engine.close()   
        return  pd.read_sql_table(table_name, con=engine1)
    
    def read_last_two(table_name):  
        engine = create_engine(conn_string)
        query = f"SELECT * FROM {table_name} ORDER BY time ASC LIMIT 2;"
        return pd.read_sql_query(query, con=engine)
    def read_last_hundred(table_name):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT * FROM {table_name} ORDER BY time ASC LIMIT 500;"
        return pd.read_sql_query(query, con=engine)
    def read_last_hundred_sh(table_name,asset,string_asset,level,today_date,today_date_start,count=500):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time,{string_asset}_{asset}_{level}_ltp,{string_asset}_{asset}_{level}_qty,{string_asset}_{asset}_{level}_avgprice,{string_asset}_{asset}_{level}_bidprice,{string_asset}_{asset}_{level}_bidqty,{string_asset}_{asset}_{level}_offprice,{string_asset}_{asset}_{level}_offqty,{string_asset}_{asset}_{level}_yrhigh,{string_asset}_{asset}_{level}_yrlow,{string_asset}_{asset}_{level}_rschange,{string_asset}_{asset}_{level}_oichange,{string_asset}_{asset}_{level}_oidiff,{string_asset}_{asset}_{level}_currentoi,{string_asset}_{asset}_{level}_totalbuyqty,{string_asset}_{asset}_{level}_totalsellqty,{string_asset}_{asset}_{level}_oihigh,{string_asset}_{asset}_{level}_oilow  FROM {table_name}  WHERE DATE(time) BETWEEN '{today_date_start}' AND '{today_date}' ORDER BY time DESC LIMIT {count};"
                
        return pd.read_sql_query(query, con=engine)
    
    def read_last_hundred_sh_fut(table_name,level,today_date,today_date_start,count=500):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time,{level}ltp,{level}qty,{level}avgprice,{level}bidprice,{level}bidqty,{level}offprice,{level}offqty,{level}yrhigh,{level}yrlow,{level}rschange,{level}oichange,{level}oidiff,{level}currentoi,{level}totalbuyqty,{level}totalsellqty,{level}oihigh,{level}oilow  FROM {table_name}  WHERE DATE(time) BETWEEN '{today_date_start}' AND '{today_date}' ORDER BY time DESC LIMIT {count};"
                
        return pd.read_sql_query(query, con=engine)
    def read_last_hundred_sh_cash(table_name,asset,today_date,today_date_start,count=500):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time,nc_{asset}_ltp FROM {table_name} WHERE DATE(time) = '{today_date}' ORDER BY time DESC LIMIT {count};"
                
        return pd.read_sql_query(query, con=engine)
    
    def read_last_hundred_sh_cash_test(table_name,asset,today_date,today_date_start,count=500):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time,nc_{asset}_ltp FROM {table_name}  WHERE DATE(time) BETWEEN '{today_date_start}' AND '{today_date}' ORDER BY time DESC LIMIT {count};"
                
        return pd.read_sql_query(query, con=engine)
    
    def read_last_one(table_name):  
        engine = create_engine(conn_string)
        query = f"SELECT * FROM {table_name} ORDER BY time DESC LIMIT 1;"
        return pd.read_sql_query(query, con=engine)
    
    def read_cal_pe(table_name):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time, opti_pe_1_oidiff,opti_pe_2_oidiff,opti_pe_3_oidiff,opti_pe_4_oidiff,opti_pe_5_oidiff,opti_pe_6_oidiff,opti_pe_7_oidiff,opti_pe_8_oidiff,opti_pe_9_oidiff,opti_pe_10_oidiff FROM {table_name};"
        return pd.read_sql_query(query, con=engine)
    
    def read_cal_ce(table_name):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time, opti_ce_1_oidiff,opti_ce_2_oidiff,opti_ce_3_oidiff,opti_ce_4_oidiff,opti_ce_5_oidiff,opti_ce_6_oidiff,opti_ce_7_oidiff,opti_ce_8_oidiff,opti_ce_9_oidiff,opti_ce_10_oidiff FROM {table_name};"
        return pd.read_sql_query(query, con=engine)
    
    def read_cal_fut(table_name):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time, fut1_oidiff,fut2_oidiff,fut3_oidiff FROM {table_name};"
        return pd.read_sql_query(query, con=engine)
    
    def read_cal_vol_ce(table_name):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time, opti_ce_1_qty,opti_ce_2_qty,opti_ce_3_qty,opti_ce_4_qty,opti_ce_5_qty,opti_ce_6_qty,opti_ce_7_qty,opti_ce_8_qty,opti_ce_9_qty,opti_ce_10_qty FROM {table_name} ;"
        return pd.read_sql_query(query, con=engine)
    
    def read_cal_vol_pe(table_name):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time, opti_pe_1_qty,opti_pe_2_qty,opti_pe_3_qty,opti_pe_4_qty,opti_pe_5_qty,opti_pe_6_qty,opti_pe_7_qty,opti_pe_8_qty,opti_pe_9_qty,opti_pe_10_qty FROM {table_name};"
        return pd.read_sql_query(query, con=engine)

    def read_cal_vol_fut(table_name):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time, fut1_qty,fut2_qty,fut3_qty FROM {table_name};"
        return pd.read_sql_query(query, con=engine)
    
    def read_last_one_eoh_ce(table_name):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time, opti_ce_1_oidiff,opti_ce_2_oidiff,opti_ce_3_oidiff,opti_ce_4_oidiff,opti_ce_5_oidiff,opti_ce_6_oidiff,opti_ce_7_oidiff,opti_ce_8_oidiff,opti_ce_9_oidiff,opti_ce_10_oidiff FROM {table_name}  ORDER BY time DESC LIMIT 2;"
        return pd.read_sql_query(query, con=engine)
    
    def read_last_one_eoh_pe(table_name):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time,  opti_pe_1_oidiff,opti_pe_2_oidiff,opti_pe_3_oidiff,opti_pe_4_oidiff,opti_pe_5_oidiff,opti_pe_6_oidiff,opti_pe_7_oidiff,opti_pe_8_oidiff,opti_pe_9_oidiff,opti_pe_10_oidiff  FROM {table_name}  ORDER BY time DESC LIMIT 2;"
        return pd.read_sql_query(query, con=engine)
    
    """bid/ask"""
    
    def read_cal_ce_bidPrice(table_name):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time, opti_ce_1_bidprice,opti_ce_2_bidprice,opti_ce_3_bidprice,opti_ce_4_bidprice,opti_ce_5_bidprice,opti_ce_6_bidprice,opti_ce_7_bidprice,opti_ce_8_bidprice,opti_ce_9_bidprice,opti_ce_10_bidprice FROM {table_name} ;"
        return pd.read_sql_query(query, con=engine)
    
    def read_cal_ce_bidQty(table_name):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time, opti_ce_1_bidqty,opti_ce_2_bidqty,opti_ce_3_bidqty,opti_ce_4_bidqty,opti_ce_5_bidqty,opti_ce_6_bidqty,opti_ce_7_bidqty,opti_ce_8_bidqty,opti_ce_9_bidqty,opti_ce_10_bidqty FROM {table_name} ;"
        return pd.read_sql_query(query, con=engine)
    
    def read_cal_ce_offPrice(table_name):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time, opti_ce_1_offprice,opti_ce_2_offprice,opti_ce_3_offprice,opti_ce_4_offprice,opti_ce_5_offprice,opti_ce_6_offprice,opti_ce_7_offprice,opti_ce_8_offprice,opti_ce_9_offprice,opti_ce_10_offprice FROM {table_name} ;"
        return pd.read_sql_query(query, con=engine)
    
    def read_cal_ce_offQty(table_name):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time, opti_ce_1_offqty,opti_ce_2_offqty,opti_ce_3_offqty,opti_ce_4_offqty,opti_ce_5_offqty,opti_ce_6_offqty,opti_ce_7_offqty,opti_ce_8_offqty,opti_ce_9_offqty,opti_ce_10_offqty FROM {table_name} ;"
        return pd.read_sql_query(query, con=engine)
    
    def read_cal_pe_bidPrice(table_name):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time, opti_pe_1_bidprice,opti_pe_2_bidprice,opti_pe_3_bidprice,opti_pe_4_bidprice,opti_pe_5_bidprice,opti_pe_6_bidprice,opti_pe_7_bidprice,opti_pe_8_bidprice,opti_pe_9_bidprice,opti_pe_10_bidprice FROM {table_name} ;"
        return pd.read_sql_query(query, con=engine)
    
    def read_cal_pe_bidQty(table_name):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time, opti_pe_1_bidqty,opti_pe_2_bidqty,opti_pe_3_bidqty,opti_pe_4_bidqty,opti_pe_5_bidqty,opti_pe_6_bidqty,opti_pe_7_bidqty,opti_pe_8_bidqty,opti_pe_9_bidqty,opti_pe_10_bidqty FROM {table_name} ;"
        return pd.read_sql_query(query, con=engine)
    
    def read_cal_pe_offPrice(table_name):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time, opti_pe_1_offprice,opti_pe_2_offprice,opti_pe_3_offprice,opti_pe_4_offprice,opti_pe_5_offprice,opti_pe_6_offprice,opti_pe_7_offprice,opti_pe_8_offprice,opti_pe_9_offprice,opti_pe_10_offprice FROM {table_name} ;"
        return pd.read_sql_query(query, con=engine)
    
    def read_cal_pe_offQty(table_name):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time, opti_pe_1_offqty,opti_pe_2_offqty,opti_pe_3_offqty,opti_pe_4_offqty,opti_pe_5_offqty,opti_pe_6_offqty,opti_pe_7_offqty,opti_pe_8_offqty,opti_pe_9_offqty,opti_pe_10_offqty FROM {table_name} ;"
        return pd.read_sql_query(query, con=engine)

    
    def read_cal_fut_bidPrice(table_name):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time, fut1_bidprice,fut2_bidprice,fut3_bidprice FROM {table_name};"
        return pd.read_sql_query(query, con=engine)

    def read_cal_fut_bidQty(table_name):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time, fut1_bidqty,fut2_bidqty,fut3_bidqty FROM {table_name};"
        return pd.read_sql_query(query, con=engine)
    
    def read_cal_fut_offPrice(table_name):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time, fut1_offprice,fut2_offprice,fut3_offprice FROM {table_name};"
        return pd.read_sql_query(query, con=engine)

    def read_cal_fut_offQty(table_name):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time, fut1_offqty,fut2_offqty,fut3_offqty FROM {table_name};"
        return pd.read_sql_query(query, con=engine)
    
    def read_cal_common(table_name, tags):  
        engine = create_engine(conn_string)
        list_as_string = ', '.join(map(str, tags))
        tag_columns = list_as_string 
        query = f"SELECT DISTINCT time, {tag_columns} FROM {table_name};"
        return pd.read_sql_query(query, con=engine)


    def read_cal_tensor_pe(table_name,date_input):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time,opti_pe_1_ltp,fut1_oidiff,fut1_ltp, opti_ce_1_qty,opti_ce_2_qty,opti_ce_3_qty,opti_ce_4_qty,opti_ce_5_qty,opti_ce_6_qty,opti_ce_7_qty,opti_ce_8_qty,opti_ce_9_qty,opti_ce_10_qty,opti_ce_1_oidiff,opti_ce_2_oidiff,opti_ce_3_oidiff,opti_ce_4_oidiff,opti_ce_5_oidiff,opti_ce_6_oidiff,opti_ce_7_oidiff,opti_ce_8_oidiff,opti_ce_9_oidiff,opti_ce_10_oidiff FROM {table_name} WHERE {date_input} ;"
        return pd.read_sql_query(query, con=engine)
    
    def read_cal_tensor_ce(table_name,date_input):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time,opti_ce_1_ltp,fut1_oidiff,fut1_ltp, opti_pe_1_qty,opti_pe_2_qty,opti_pe_3_qty,opti_pe_4_qty,opti_pe_5_qty,opti_pe_6_qty,opti_pe_7_qty,opti_pe_8_qty,opti_pe_9_qty,opti_pe_10_qty,opti_pe_1_oidiff,opti_pe_2_oidiff,opti_pe_3_oidiff,opti_pe_4_oidiff,opti_pe_5_oidiff,opti_pe_6_oidiff,opti_pe_7_oidiff,opti_pe_8_oidiff,opti_pe_9_oidiff,opti_pe_10_oidiff FROM {table_name} WHERE {date_input} ;"
        return pd.read_sql_query(query, con=engine)
    
    
    
    def read_cal_tensor_pe_test(table_name,date_input,cont):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time,opti_pe_{cont}_ltp,fut1_oidiff,fut1_ltp, opti_pe_{cont}_qty,opti_pe_{cont}_oidiff FROM {table_name} WHERE {date_input} ;"
        return pd.read_sql_query(query, con=engine)
    
    def read_cal_tensor_ce_test(table_name,date_input,cont):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time,opti_ce_{cont}_ltp,fut1_oidiff,fut1_ltp, opti_ce_{cont}_qty,opti_ce_{cont}_oidiff FROM {table_name} WHERE {date_input} ;"
        return pd.read_sql_query(query, con=engine)
    
        
    def read_last_hundred_maxima(table_name,today_date,today_date_start,count=500):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time,opti_pe_1_ltp,opti_pe_2_ltp,opti_pe_3_ltp,opti_pe_4_ltp,opti_pe_5_ltp,opti_pe_6_ltp,opti_pe_7_ltp,opti_pe_8_ltp,opti_pe_9_ltp,opti_pe_10_ltp,opti_pe_11_ltp,fut1_ltp,fut2_ltp,fut3_ltp  FROM {table_name}  WHERE DATE(time) BETWEEN '{today_date_start}' AND '{today_date}' ORDER BY time DESC LIMIT {count};"
        return pd.read_sql_query(query, con=engine)  

    def read_last_hundred_maxima_ce(table_name,today_date,today_date_start,count=500):  
        engine = create_engine(conn_string)
        query = f"SELECT DISTINCT time,opti_ce_1_ltp,opti_ce_2_ltp,opti_ce_3_ltp,opti_ce_4_ltp,opti_ce_5_ltp,opti_ce_6_ltp,opti_ce_7_ltp,opti_ce_8_ltp,opti_ce_9_ltp,opti_ce_10_ltp,opti_ce_11_ltp  FROM {table_name}  WHERE DATE(time) BETWEEN '{today_date_start}' AND '{today_date}' ORDER BY time DESC LIMIT {count};"
        return pd.read_sql_query(query, con=engine)     
# example
# data = database_setup.read('token')
