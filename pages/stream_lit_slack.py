# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 17:52:54 2024

@author: kuldeep.rana
"""

import logging, os, json
from base64 import b64decode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

#Update following variables before test.
SLACK_WEBHOOK_URL='https://hooks.slack.com/services/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'
SLACK_CHANNEL_NAME="grafana123"     #Exaple: '#my-reporting-slack-channel'
SLACK_USERNAME='kuldeep7322@gmail.com'    
SLACK_ICON_EMOJI=':robot_face:'

#Logging Variables
logger = logging.getLogger()
logger.setLevel(logging.INFO)

#Slack Message function
def lambda_to_slack(SLACK_MSG):
    payload = {'text':SLACK_MSG,'channel':SLACK_CHANNEL_NAME,'icon_emoji':SLACK_ICON_EMOJI,'username':SLACK_USERNAME}
    print('Sending Message to Slack')
    req = Request(SLACK_WEBHOOK_URL, json.dumps(payload).encode('utf-8'))
    try:
        response = urlopen(req)
        response.read()
        logger.info("Message posted to %s", payload['channel'])
    except HTTPError as e:
        logger.error("Request failed: %d %s", e.code, e.reason)
    except URLError as e:
        logger.error("Server connection failed: %s", e.reason)
    return 0

#Lamdba Function Code
def lambda_handler(event, context):
    # TODO implement
    lambda_to_slack(context)
    return 0
# lambda_handler("test", "test")