from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

finviz_url = 'https://finviz.com/quote.ashx?t='
tickers = ['AMZN', 'GOOG', 'FB']

news_tables = {}
for ticker in tickers:
    # Create the response object for the request URL
    url = finviz_url + ticker

    req = Request(url=url, headers={
            'user-agent': 'my-app'
    })
    response = urlopen(req)

    # Use BeautifulSoup to the scrape the response(an object) in html form
    html = BeautifulSoup(response, 'html')
    # use the news-table id from the html response to read the headings
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table

# parse using pandas, the news_tables into a required format (title and time-stamp)
parsed_data = []

for ticker, news_table in news_tables.items():

    for row in news_table.findAll('tr'):

        title = row.a.get_text()
        date_data = row.td.text.split(' ')

        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]

        parsed_data.append([ticker, date, time, title])

# Applying sentiment analysis using nltk vader on the titles in parsed_data
# pip install nltk => go to console => run> nltk.download() => download vader from all packages
# Using pandas to manipulate data
df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'title'])
vader = SentimentIntensityAnalyzer()
f = lambda title: vader.polarity_scores(title)['compound']
df['compound'] = df['title'].apply(f)
# Update the date to datetime format
df['date'] = pd.to_datetime(df.date).dt.date

# Visualizing the data with matplotlib
plt.figure(figsize=(10, 8))
mean_df = df.groupby(['ticker', 'date']).mean()
# to have the data as a chart-like structure, we will first unstack it, to put date on x-axis, xs: cross-section
mean_df = mean_df.unstack()
mean_df = mean_df.xs('compound', axis="columns").transpose()
mean_df.plot(kind='bar')
plt.show()










