"""
Basic sentiment analysis test using Reddit news headlines from r/politics
and NLTK's Vader Sentiment Analyzer to get an idea of how sentiment analysis works,
plotting negative/positive distribution, and working with tokenizers/stopwords.

Resource:
https://www.learndatasci.com/tutorials/sentiment-analysis-reddit-headlines-pythons-nltk/
"""

from IPython import display
import math
from pprint import pprint
import pandas as pd
import nltk
# nltk.download('vader_lexicon')
import matplotlib.pyplot as plt
import seaborn as sns
import praw
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
from nltk.tokenize import word_tokenize, RegexpTokenizer
# nltk.download('stopwords')
from nltk.corpus import stopwords

# use Reddit API wrapper to loop through r/politics subreddit headlines
reddit = praw.Reddit(client_id='_QfuGGMnju3QWg',
                     client_secret='nd1BKP7gvZxSf0zCe6d8FrqPUxQ',
                     user_agent='rc3109')
# remove duplicates
headlines = set()

# loads newest 1000 headlines
for submission in reddit.subreddit('politics').new(limit=1000):
    headlines.add(submission.title)
    display.clear_output()
    # print(len(headlines))

# test using NLTK's Vader Sentiment Analyzer
sia = SIA()
results = []

for line in headlines:
    pol_score = sia.polarity_scores(line)
    pol_score['headline'] = line
    results.append(pol_score)

# pprint(results[:3], width=100)

# put data into dataframe
df = pd.DataFrame.from_records(results)
# print(df.head())

# create positive and negative labels
df['label'] = 0
df.loc[df['compound'] > 0.2, 'label'] = 1
df.loc[df['compound'] < -0.2, 'label'] = -1
# print(df.head())

# print some of positive and negative headlines
df2 = df[['headline', 'label']]
df2.to_csv('reddit_headlines_labels.csv', mode='a', encoding='utf-8', index=False)

print("Positive headlines:\n")
pprint(list(df[df['label'] == 1].headline)[:5], width=200)

print("\nNegative headlines:\n")
pprint(list(df[df['label'] == -1].headline)[:5], width=200)

# check total positive and negative
# print(df.label.value_counts())
# print(df.label.value_counts(normalize=True) * 100)

# plot values
fig, ax = plt.subplots(figsize=(8, 8))
counts = df.label.value_counts(normalize=True) * 100
sns.barplot(x=counts.index, y=counts, ax=ax)
ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.set_ylabel("Percentage")
plt.show()

# set tokenizer and stopwords
tokenizer = RegexpTokenizer(r'\w+')
stop_words = stopwords.words('english')

# function to perform lowercasing, tokenizing, and stopword removal
def process_text(headlines):
    tokens = []
    for line in headlines:
        toks = tokenizer.tokenize(line)
        toks = [t.lower() for t in toks if t.lower() not in stop_words]
        tokens.extend(toks)

    return tokens

# find positive words
pos_lines = list(df[df.label == 1].headline)
pos_tokens = process_text(pos_lines)
pos_freq = nltk.FreqDist(pos_tokens)
print(pos_freq.most_common(20))
# plot frequency distribution for most positive word
y_val = [x[1] for x in pos_freq.most_common()]
fig = plt.figure(figsize=(10,5))
plt.plot(y_val)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Word Frequency Distribution (Positive)")
plt.show()
# plot with log-log
y_final = []
for i, k, z, t in zip(y_val[0::4], y_val[1::4], y_val[2::4], y_val[3::4]):
    y_final.append(math.log(i + k + z + t))
x_val = [math.log(i + 1) for i in range(len(y_final))]
fig = plt.figure(figsize=(10,5))
plt.xlabel("Words (Log)")
plt.ylabel("Frequency (Log)")
plt.title("Word Frequency Distribution (Positive)")
plt.plot(x_val, y_final)
plt.show()


# find negative words
neg_lines = list(df2[df2.label == -1].headline)
neg_tokens = process_text(neg_lines)
neg_freq = nltk.FreqDist(neg_tokens)
print(neg_freq.most_common(20))
# plot frequency distribution for most negative word
y_val = [x[1] for x in neg_freq.most_common()]
fig = plt.figure(figsize=(10,5))
plt.plot(y_val)
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.title("Word Frequency Distribution (Negative)")
plt.show()
# plot with log-log
y_final = []
for i, k, z in zip(y_val[0::3], y_val[1::3], y_val[2::3]):
    if i + k + z == 0:
        break
    y_final.append(math.log(i + k + z))
x_val = [math.log(i+1) for i in range(len(y_final))]
fig = plt.figure(figsize=(10,5))
plt.xlabel("Words (Log)")
plt.ylabel("Frequency (Log)")
plt.title("Word Frequency Distribution (Negative)")
plt.plot(x_val, y_final)
plt.show()