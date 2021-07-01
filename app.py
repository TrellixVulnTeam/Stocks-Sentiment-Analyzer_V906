import torch
import flair
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import base64
import requests
from datetime import datetime, timedelta
import re

sentiment_model = flair.models.TextClassifier.load('en-sentiment')
print("Sentiment Flair Model Generated \n")

# Setting Twitter API, with api key and api security keys-----------------------
client_key = 'TxUjAU6auFnrhhiu8y12upAMh'
client_secret = '3FNeTewgI9IlRgqj2wRSv2QVGMNSNuOH1wXaGdApKwIwdQY1LF'

key_secret = '{}:{}'.format(client_key, client_secret).encode('ascii')
b64_encoded_key = base64.b64encode(key_secret)
b64_encoded_key = b64_encoded_key.decode('ascii')

base_url = 'https://api.twitter.com/'
auth_url = '{}oauth2/token'.format(base_url)

auth_headers = {
    'Authorization': 'Basic {}'.format(b64_encoded_key),
    'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'
}
auth_data = {
    'grant_type': 'client_credentials'
}
auth_resp = requests.post(auth_url, headers=auth_headers, data=auth_data)

auth_resp.status_code

auth_resp.json().keys()

access_token = auth_resp.json()['access_token']
# Twitter API and Token Generator-----------------------------------------------
print("API Tokens Done \n")


st.write("""
# Stocks Prediction App
This Web App analyzes the behaviour of the input stock on the basis of sentiments of past hour tweets trained in a Flair Model. You can find the entire source code for the models and web deployment [here](https://github.com/BugBear646/Stocks-Sentiment-Analyzer/).
Feel free to fork and contribute.😀
""")
st.sidebar.header('User Input Stock')

def user_input_features():
    input_item = st.sidebar.selectbox('Please select a Stock',('MSFT','AMZN','TSLA', 'GOOGL', 'IBM', 'FB', 'AAPL', 'NFLX'))
    return input_item
input_stock = user_input_features()


# Take Input from User----------------------------------------------------------
#print(" Please Type the Stock Name you want to predict \n")
#input_stock = input()

search_headers = {
    'Authorization': 'Bearer {}'.format(access_token)
}

search_params = {
    'q': input_stock,
    'tweet_mode': 'extended',
    'lang': 'en',
    'count': '100'
}

search_url = '{}1.1/search/tweets.json'.format(base_url)


# Returning Data from Twitter---------------------------------------------------
def get_data(tweet):
    data = {
        'id': tweet['id_str'],
        'created_at': tweet['created_at'],
        'text': tweet['full_text']
    }
    return data

df = pd.DataFrame() #Initiating the dataframe


#You can use this if you want to perform time data
#
# Making Time Based Inputs------------------------------------------------------
#dtformat = '%Y-%m-%dT%H:%M:%SZ'  # the date format string required by twitter

# we use this function to subtract 60 mins from our datetime string
#def time_travel(now, mins):
    #now = datetime.strptime(now, dtformat)
    #back_in_time = now - timedelta(minutes=mins)
    #return back_in_time.strftime(dtformat)

#now = datetime.now()  # get the current datetime, this is our starting point
#last_week = now - timedelta(days=7)  # datetime one week ago = the finish line
#now = now.strftime(dtformat)  # convert now datetime to format for API


#print("Dataframing Started \n")
# Scrapping tweets and pushing it to the DataFrame -----------------------------

#df = pd.DataFrame() #Initiating the dataframe
#while True:
    #if datetime.strptime(now, dtformat) < last_week:
        # if we have reached 7 days ago, break the loop
        #break
    #pre60 = time_travel(now, 60)  # get 60 minutes before 'now'
    # assign from and to datetime parameters for the API
    #search_params['start_time'] = pre60
    #search_params['end_time'] = now
    #search_resp = requests.get(search_url, headers=search_headers, params=search_params) # send the request
    #now = pre60  # move the window 60 minutes earlier
    # iteratively append our tweet data to our dataframe

search_resp = requests.get(search_url, headers=search_headers, params=search_params) # send the request
for tweet in search_resp.json()['statuses']:
    row = get_data(tweet)
    df = df.append(row, ignore_index=True)

#print("Dataframing Done! \n")
# Regex-------------------------------------------------------------------------
def clean(tweet):
    whitespace = re.compile(r"\s+")
    web_address = re.compile(r"(?i)http(s):\/\/[a-z0-9.~_\-\/]+")
    user = re.compile(r"(?i)@[a-z0-9_]+")

    # we then use the sub method to replace anything matching
    tweet = whitespace.sub(' ', str(tweet))
    tweet = web_address.sub('', str(tweet))
    tweet = user.sub('', str(tweet))

#print("Prob Started")
# The Sentiments and Probabilities appending------------------------------------
# we will append probability and sentiment preds later
probs = []
sentiments = []

# use regex expressions (in clean function) to clean tweets
clean(df['text'])

for tweet in df['text']:
    if tweet.strip() == "":
       probs.append("")
       sentiments.append("")

    else:
      # make prediction
      sentence = flair.data.Sentence(tweet)
      sentiment_model.predict(sentence)
      # extract sentiment prediction
      if(sentence.labels[0].value=='NEGATIVE'):
        probs.append(-1*sentence.labels[0].score)  # numerical score 0-1
      else:
         probs.append(sentence.labels[0].score)  # numerical score 0-1
      sentiments.append(sentence.labels[0].value)  # 'POSITIVE' or 'NEGATIVE'

# add probability and sentiment predictions to tweets dataframe
df['probability'] = probs
df['sentiment'] = sentiments

#print("prob Done ! \n")
df=df.sort_values("created_at")

#print(df.shape[0])
y=df['sentiment'].value_counts()
x=df.groupby('sentiment')['probability'].sum()

Positive_Score = x[1]/y[0]
#print(Positive_Score)
Negative_Score = x[0]/y[1]
#print(Negative_Score)

st.write("""
Representation of Sentiments of the Last 100 Tweets related to the Input Stock.
""")
Prediction_Score = x[1]/(-1*x[0])
print(Prediction_Score)

#df = df.rename(columns={'created_at':'index'}).set_index('index')
st.area_chart(df['probability'])
if st.button("Predict"):
    st.success("The Probability of Positive Sentiments: "+str(Positive_Score))
    st.success("The Probability of Negative Sentiments: "+str(Negative_Score))

    # Pie chart, where the slices will be ordered and plotted counter-clockwise
    labels = ['Positive Score','Negative Score']
    values = [x[1] , -1*x[0] ]

    # pull is given as a fraction of the pie radius
    fig3 = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0, 0, 0.2, 0])])
    fig3.update_layout(title="<b>Input Stock Current Twitter Sentiment Analysis</b>")
    st.plotly_chart(fig3)

    st.success("The stock has a Prediction Probability Score of: "+str(Prediction_Score))
    st.write("""
    The Prediction Probability Score is a measure of how the stock is performing. If >1, then it is performing well, and if 0<Prediction Probability Score<1 then the performace is poor.
    The more greater it is from 1, the more better it is performing and vice versa.
    """)
