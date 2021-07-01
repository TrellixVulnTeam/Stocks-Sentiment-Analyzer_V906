# Import libraries
import nltk
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from urllib.request import urlopen, Request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import torch
import flair
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import base64
import requests
from datetime import datetime, timedelta
import re
import plotly.express as px

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
# Stalker App 🧐
This Web App analyzes the behaviour of the stocks on the basis of sentiments of past hour tweets trained in a Flair Model and past one week news headlines. You can find the entire source code for the models and web deployment [here](https://github.com/BugBear646/Stocks-Sentiment-Analyzer/).
Feel free to fork and contribute.😀
""")
st.sidebar.header('User Input Stocks')

def user_input_features():
    input_stock = st.sidebar.selectbox('Input Stock',('MSFT','AMZN', 'AAPL', 'GOOGL', 'IBM', 'FB', 'TSLA', 'NFLX'))
    comp_1 = st.sidebar.selectbox('Competitor Stock 1',('AMZN','AAPL', 'GOOGL', 'IBM', 'FB', 'TSLA', 'NFLX', 'MSFT'))
    comp_2 = st.sidebar.selectbox('Competitor Stock 2',('AAPL', 'GOOGL', 'IBM', 'FB', 'TSLA', 'NFLX', 'MSFT','AMZN'))
    data = {'input_stock': input_stock,
            'comp_1': comp_1,
            'comp_2': comp_2}
    features = pd.DataFrame(data, index=[0])
    return features

inp_df = user_input_features()
inp_df = inp_df[:1] # Selects only the first row (the user input data)

input_stock = inp_df['input_stock'][0]
comp_1 = inp_df['comp_1'][0]
comp_2 = inp_df['comp_2'][0]
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
#print(Prediction_Score)

#df = df.rename(columns={'created_at':'index'}).set_index('index')
st.area_chart(df['probability'])
if st.button("Predict"):
    st.write("""
    Predictions of the Input Stock:
    """)
    # Pie chart, where the slices will be ordered and plotted counter-clockwise
    labels = ['Positive Score','Negative Score']
    values = [x[1] , -1*x[0] ]

    # pull is given as a fraction of the pie radius
    fig3 = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0, 0, 0.2, 0])])
    fig3.update_layout(title="<b>Input Stock Current Twitter Sentiment Analysis</b>")
    st.plotly_chart(fig3)

    st.success("The Probability of Positive Sentiments: "+str(Positive_Score))
    st.success("The Probability of Negative Sentiments: "+str(Negative_Score))
    st.success("The stock has a Prediction Probability Score of: "+str(Prediction_Score))
    st.write("""
    The Prediction Probability Score is a measure of how the stock is performing. If >1, then it is performing well, and if 0<Prediction Probability Score<1 then the performace is poor.
    The more greater it is from 1, the more better it is performing and vice versa.
    """)

print("Input Stock Predictions are done, give me just a few seconds for the competitor stocks. \n")

# Parameters
n = 3 #the # of article headlines displayed per ticker
#print("Input 3 stocks, first your target stock and later stocks you want to compare it with: ")
#input_stock=input()
#comp_1=input()
#comp_2=input()
tickers = [input_stock, comp_1, comp_2]

# Get Data
finwiz_url = 'https://finviz.com/quote.ashx?t='
news_tables = {}

for ticker in tickers:
    url = finwiz_url + ticker
    req = Request(url=url,headers={'user-agent': 'my-app/0.0.1'})
    resp = urlopen(req)
    html = BeautifulSoup(resp, features="lxml")
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table

try:
    for ticker in tickers:
        df = news_tables[ticker]
        df_tr = df.findAll('tr')

        #print ('\n')
        #print ('Recent News Headlines for {}: '.format(ticker))

        for i, table_row in enumerate(df_tr):
            a_text = table_row.a.text
            td_text = table_row.td.text
            td_text = td_text.strip()
            #print(a_text,'(',td_text,')')
            if i == n-1:
                break
except KeyError:
    pass


# Iterate through the news
parsed_news = []
for file_name, news_table in news_tables.items():
    for x in news_table.findAll('tr'):
        text = x.a.get_text()
        date_scrape = x.td.text.split()

        if len(date_scrape) == 1:
            time = date_scrape[0]

        else:
            date = date_scrape[0]
            time = date_scrape[1]

        ticker = file_name.split('_')[0]
        parsed_news.append([ticker, date, time, text])




# Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()
columns = ['Ticker', 'Date', 'Time', 'Headline']
news = pd.DataFrame(parsed_news, columns=columns)
scores = news['Headline'].apply(analyzer.polarity_scores).tolist()

df_scores = pd.DataFrame(scores)
news = news.join(df_scores, rsuffix='_right')

# View Data
news['Date'] = pd.to_datetime(news.Date).dt.date

unique_ticker = news['Ticker'].unique().tolist()
news_dict = {name: news.loc[news['Ticker'] == name] for name in unique_ticker}

values = []
for ticker in tickers:
    dataframe = news_dict[ticker]
    dataframe = dataframe.set_index('Ticker')
    dataframe = dataframe.drop(columns = ['Headline'])
    #print ('\n')
    #print (dataframe.head())

    mean = round(dataframe['compound'].mean(), 2)
    values.append(mean)

df1 = pd.DataFrame(list(zip(tickers, values)), columns =['Ticker', 'Mean_Sentiment'])
df1 = df1.set_index('Ticker')
#print ('\n')
#print(df1.shape)
#print (df1)
print("Scrapped everything, lemme bring my drawing kit so that I can plot the comparisons :) \n")


if st.button("Compare"):
    # Pie chart, where the slices will be ordered and plotted counter-clockwise
    # This dataframe has 244 lines, but 4 distinct values for `day`
    #df = px.data.tips()
        random_x = [df1['Mean_Sentiment'][0], df1['Mean_Sentiment'][1], df1['Mean_Sentiment'][2]]
        names = [input_stock, comp_1, comp_2]

        fig = go.Figure([go.Bar(x=names, y=random_x)])
        fig.update_layout(title="<b>Input Stock vs Competitor Stocks</b>")
        st.plotly_chart(fig)
        st.write("""
        The above is a plot of the sentiments' score of Stocks, scrapped from past one week News headlines.
        """)

print ("I hope you enjoyed, feel free to contribute. <3 \n")
print ("Press <Ctrl + C> to STOP!")
