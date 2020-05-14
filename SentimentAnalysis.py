import pandas as pd
import flair
from textblob import TextBlob
import os
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# initialize Flair and VADER models
flair_sentiment = flair.models.TextClassifier.load('en-sentiment')
fmt = '%Y-%m-%d %H:00:00'
sid = SentimentIntensityAnalyzer()

# get sentiment value from Flair
def get_sentiment_val_for_flair(sentiments):
    total_sentiment = str(sentiments)
    neg = 'NEGATIVE' in total_sentiment
    if neg:
        total_sentiment = total_sentiment.replace('NEGATIVE', '')
    else:
        total_sentiment = total_sentiment.replace('POSITIVE', '')

    total_sentiment = total_sentiment.replace('(', '').replace('[', '').replace(')', '').replace(']', '')

    val = float(total_sentiment)
    if neg:
        return -val
    return val

# performs sentiment analysis using Flair, VADER, and TextBlob
def get_sentiment_report(input_filename, output_filename):
    # pre-processing data
    df = pd.read_csv(input_filename, encoding="ISO-8859-1")
    df = df[['title', 'selftext', 'publish_date']]
    df = df.fillna('')
    df['text'] = df['title'] + ' ' + df['selftext']
    df.set_index('publish_date', inplace=True)
    df.drop(['title', 'selftext'], axis=1, inplace=True)

    # calculate sentiment values and add to dataframe
    for row_i, row in df.iterrows():
        tb_sentiment_polarity_dict = dict()
        tb_sentiment_subjectivity_dict = dict()
        flair_sentiment_dict = dict()

        sid_pos_dict = dict()
        sid_neg_dict = dict()
        sid_neu_dict = dict()
        sid_com_dict = dict()

        # Flair values
        data = row['text']
        print(row_i)
        print(data[0:15])
        flair_s = flair.data.Sentence(data)
        flair_sentiment.predict(flair_s)
        flair_total_sentiment = flair_s.labels
        flair_val = get_sentiment_val_for_flair(flair_total_sentiment)
        flair_sentiment_dict[str(row_i)] = flair_val

        # TextBlob values
        tb_sentiment_polarity_dict[str(row_i)] = TextBlob(data).sentiment[0]
        tb_sentiment_subjectivity_dict[str(row_i)] = TextBlob(data).sentiment[1]

        # VADER values
        ss = sid.polarity_scores(data)
        sid_pos_dict[str(row_i)] = ss['pos']
        sid_neg_dict[str(row_i)] = ss['neg']
        sid_neu_dict[str(row_i)] = ss['neu']
        sid_com_dict[str(row_i)] = ss['compound']

        # Add to dataframe
        flair_df = pd.DataFrame.from_dict(flair_sentiment_dict, orient='index', columns=['reddit_flair'])
        flair_df.index.name = 'timestamp'
        tb_polarity_df = pd.DataFrame.from_dict(tb_sentiment_polarity_dict, orient='index', columns=['reddit_tb_polarity'])
        tb_polarity_df.index.name = 'timestamp'
        tb_subjectivity_df = pd.DataFrame.from_dict(tb_sentiment_subjectivity_dict, orient='index', columns=['reddit_tb_subjectivity'])
        tb_subjectivity_df.index.name = 'timestamp'
        sid_pos_df = pd.DataFrame.from_dict(sid_pos_dict, orient='index', columns=['reddit_sid_pos'])
        sid_pos_df.index.name = 'timestamp'
        sid_neg_df = pd.DataFrame.from_dict(sid_neg_dict, orient='index', columns=['reddit_sid_neg'])
        sid_neg_df.index.name = 'timestamp'
        sid_neu_df = pd.DataFrame.from_dict(sid_neu_dict, orient='index', columns=['reddit_sid_neu'])
        sid_neu_df.index.name = 'timestamp'
        sid_com_df = pd.DataFrame.from_dict(sid_com_dict, orient='index', columns=['reddit_sid_com'])
        sid_com_df.index.name = 'timestamp'
        final_senti_df = pd.concat([flair_df, tb_polarity_df, tb_subjectivity_df, sid_pos_df, sid_neg_df, sid_neu_df, sid_com_df], axis=1)

        if os.path.exists(output_filename):
            keep_header = False
        else:
            keep_header = True

        final_senti_df.to_csv(output_filename, mode='a', header=keep_header)

    return

# removes duplicates from final report
def clean_sentiment_report(input_filename, output_filename):
    master_df = pd.read_csv(input_filename, index_col=0)
    master_df.index = pd.to_datetime(master_df.index)
    idx = np.unique(master_df.index, return_index=True)[1]
    master_df = master_df.iloc[idx]
    master_df.to_csv(output_filename)

if __name__ == '__main__':
    input_filename = 'Biden_data.csv'
    output_sentiment_filename = 'Biden_sentiment.csv'
    get_sentiment_report(input_filename, output_sentiment_filename)