import pandas as pd
from matplotlib import pyplot as plt

# Trump_sentiment.csv
df1 = pd.read_csv("Trump_sentiment.csv")
df1.set_index(['timestamp'])

# convert to date time
df1.index = pd.to_datetime(df1.index, unit='d')

# create dataframe storing weekly summary
trump_weekly = pd.DataFrame()
trump_weekly['flair_score'] = df1.reddit_flair.resample('W').mean()
trump_weekly['vader_score'] = df1.reddit_sid_com.resample('W').mean()
trump_weekly['tb_score'] = df1.reddit_tb_polarity.resample('W').mean()
trump_weekly = trump_weekly.truncate(before='2020-01-01', after='2020-05-01')
print("------- Trump Weekly Summary -------\n", trump_weekly)

# plot values
fig, ax1 = plt.subplots()
ax1.set_title('Sentiment Analysis - Donald Trump (January - May 2020)')
ax2 = ax1.twinx()
ax1.plot(trump_weekly.index, trump_weekly['flair_score'], 'g-')
ax2.plot(trump_weekly.index, trump_weekly['vader_score'], 'b-')
ax1.set_xlabel('Date')
ax1.set_ylabel('Flair Score', color='g')
ax2.set_ylabel('Vader Score', color='b')
fig.autofmt_xdate()
plt.show()

# Biden_sentiment.csv
df2 = pd.read_csv("Biden_sentiment.csv")
df2.set_index(['timestamp'])

# convert to date time
df2.index = pd.to_datetime(df2.index, unit='d')

# create dataframe storing weekly summary
biden_weekly = pd.DataFrame()
biden_weekly['flair_score'] = df2.reddit_flair.resample('W').mean()
biden_weekly['vader_score'] = df2.reddit_sid_com.resample('W').mean()
biden_weekly['tb_score'] = df2.reddit_tb_polarity.resample('W').mean()
biden_weekly = biden_weekly.truncate(before='2020-01-01', after='2020-05-01')
print("------- Biden Weekly Summary -------\n", biden_weekly)

# plot values
fig, ax1 = plt.subplots()
ax1.set_title('Sentiment Analysis - Joe Biden (January - May 2020)')
ax2 = ax1.twinx()
ax1.plot(biden_weekly.index, biden_weekly['flair_score'], 'g-')
ax2.plot(biden_weekly.index, biden_weekly['vader_score'], 'b-')
ax1.set_xlabel('Date')
ax1.set_ylabel('Flair Score', color='g')
ax2.set_ylabel('Vader Score', color='b')
fig.autofmt_xdate()
plt.show()