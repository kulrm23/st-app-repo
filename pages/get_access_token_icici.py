# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 04:17:22 2024

@author: kuldeep.rana
"""
try:
    import urllib
    import time
    from   breeze_connect  import BreezeConnect
    from   database_setup  import database_setup
    from   json_reader     import read_config
    import pandas as pd
    from   pywinauto                       import Application 
    from   urllib.parse                    import urlparse, parse_qs, unquote
    import json
    import webbrowser
    import pandas as pd

except OSError as err:
    print(err)
    
url1        = "https://api.icicidirect.com/apiuser/login?api_key=W57%5e039330%60163143%60w385Ug8404ORL1"
chrome_path = 'C:/Program Files/Google/Chrome/Application/chrome.exe %s'

webbrowser.get(chrome_path).open(url1)

time.sleep(40)

app    = Application(backend='uia')

app.connect(title_re=".*Chrome.*")

element_name        = "Address and search bar"
dlg                 = app.top_window()
url1                = dlg.child_window(title=element_name, control_type="Edit").get_value()
parsed_url          = urlparse(url1)
query_params        = parse_qs(parsed_url.query)
request_token_value = query_params.get('apisession', [])[0] if 'apisession' in query_params else None
request_token       = pd.DataFrame(pd.Series(request_token_value, name='token'))   

config = read_config(config_file="config.json", readjson_with_comments=True) 

key_to_replace = 'session_token'
new_value = int(request_token['token'].iloc[0])

config[key_to_replace] = new_value
filepath = "config.json"


with open(filepath,'w') as f:
    json.dump(config,f,indent=4) 
    
filepath1 = "E:\st1\st1\config.json"
    
with open(filepath1,'w') as f:
    json.dump(config,f,indent=4) 
        
    
    
database_setup.write_o('token_ici', request_token)
# 33027783

