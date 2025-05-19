# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 04:39:06 2024

@author: kuldeep.rana
"""
import pandas as pd

class Zone_finder_custom():
    def levels_cal(self,df,fibScale):

        fib_levels = (0.24, 0.38, 0.82, 1.68, 2.3, 3.6, 4.0)
    
        fibLevel1 = fib_levels[0]
        fibLevel2 = fib_levels[1]
        fibLevel3 = fib_levels[2]
        fibLevel4 = fib_levels[3]
        fibLevel5 = fib_levels[4]
        fibLevel6 = fib_levels[5]
        fibLevel7 = fib_levels[6]
        
        df.reset_index(inplace=True)
        df['datetime_column'] = pd.to_datetime(df['Datetime'])

        today = pd.Timestamp.now().date() 
        today_data = df[df['datetime_column'].dt.date == today]
        price_fut_high = today_data['high']
        price_fut_low  = today_data['low']  
        
        price_5_max    = max(price_fut_high.head(1))
        price_5_min    = min(price_fut_low.head(1))
        price_difference = price_5_max - price_5_min
        # print(price_difference)
        # if price_difference >= 10:
        #     price_5_max =  min(price_fut_high.head(5))
        #     price_5_min =  max(price_fut_low.head(5))
        # else:
        #     price_5_max    = max(price_fut_high.head(5))
        #     price_5_min    = min(price_fut_low.head(5))
        
        range_3_Low    = price_5_min - ((price_5_max-price_5_min) * fibLevel3 * fibScale)
        range_6_Low    = price_5_min - ((price_5_max-price_5_min) * fibLevel2 * fibScale)
        range_3_High   = price_5_min + ((price_5_max-price_5_min) * fibLevel3 * fibScale)
        range_6_High   = price_5_min + ((price_5_max-price_5_min) * fibLevel2 * fibScale)
        range_8_Low    = price_5_min - ((price_5_max-price_5_min) * fibLevel4 * fibScale)
        range_9_Low    = price_5_min - ((price_5_max-price_5_min) * fibLevel5 * fibScale)
        range_8_High   = price_5_min + ((price_5_max-price_5_min) * fibLevel4 * fibScale)
        range_9_High   = price_5_min + ((price_5_max-price_5_min) * fibLevel5 * fibScale)
        range_7_Low    = price_5_min - ((price_5_max-price_5_min) * fibLevel4 * fibScale)
        range_10_Low   = price_5_min - ((price_5_max-price_5_min) * fibLevel5 * fibScale)
        range_7_High   = price_5_min + ((price_5_max-price_5_min) * fibLevel4 * fibScale)
        range_10_High  = price_5_min + ((price_5_max-price_5_min) * fibLevel5 * fibScale)
        range_1_Low    = price_5_min - ((price_5_max-price_5_min) * fibLevel1 * fibScale)
        range_1_high   = price_5_min + ((price_5_max-price_5_min) * fibLevel1 * fibScale)
        range_11_Low   = price_5_min - ((price_5_max-price_5_min) * fibLevel6 * fibScale)
        range_11_High  = price_5_min + ((price_5_max-price_5_min) * fibLevel6 * fibScale)
        range_12_Low   = price_5_min - ((price_5_max-price_5_min) * fibLevel7 * fibScale)
        range_12_High  = price_5_min + ((price_5_max-price_5_min) * fibLevel7 * fibScale)
        
        fib_df_main    = [range_3_Low,range_6_Low,range_3_High,range_6_High,
                          range_8_Low,range_9_Low,range_8_High,range_9_High,
                          range_7_Low,range_10_Low,range_7_High,range_10_High,
                          range_1_Low,range_1_high,
                          range_11_Low,range_12_Low,range_11_High,range_12_High]
        

        return fib_df_main
    def plots(self,axn,price,fib_df_main):
        
       axn.plot(price, label='Actual Values', marker='o', color='blue', linestyle='-', markersize=1, alpha=1, linewidth=1)
       axn.fill_between(price.index, fib_df_main[0],  fib_df_main[1], color='red',    alpha=0.2, label='Fib Region 1')
       axn.fill_between(price.index, fib_df_main[2],  fib_df_main[3], color='green',  alpha=0.2, label='Fib Region 2')
       axn.fill_between(price.index, fib_df_main[4],  fib_df_main[5], color='blue',   alpha=0.2, label='Fib Region 3')
       axn.fill_between(price.index, fib_df_main[6],  fib_df_main[7], color='blue',   alpha=0.2, label='Fib Region 4')
       axn.fill_between(price.index, fib_df_main[8],  fib_df_main[9], color='red',    alpha=0.2, label='Fib Region 5')
       axn.fill_between(price.index, fib_df_main[10], fib_df_main[11],color='green',  alpha=0.2, label='Fib Region 6')
       axn.fill_between(price.index, fib_df_main[16], fib_df_main[17],color='blue',   alpha=0.2, label='Fib Region 7')
       axn.fill_between(price.index, fib_df_main[15], fib_df_main[14],color='blue',   alpha=0.2, label='Fib Region 8')
   
       line_colors = ['red', 'red', 'green','green','red', 'red', 'green','green','red', 'red', 'green','green','blue','orange',
                       'green','green','blue','orange']
       for i, fib_value in enumerate(fib_df_main):
           axn.axhline(y=fib_value, color=line_colors[i], linestyle='-', linewidth=.5, label=f'Fib {fib_df_main[i]:.3f}')

        
