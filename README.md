# Political Sentiment Analyzer
University of Colorado Denver | CSCI4930 - Final Project | Ryan Cheng

Project Overview:
BasicSA.py - basic test using Vader SA on 1000 headlines, view negative/positive word distribution and tokenizers/stopwords
RetrieveData.py - retrieves data from Reddit using Pushshift API
SentimentAnalysis.py - performs sentiment analysis on data using Flair, Vader, and TextBlob models, outputs results from each model
PlotResults.py - plots results to show trend in sentiment over time

reddit_headlines_labels.csv - sentiment labels using VADER from BasicSA.py
Trump_data.csv - 35,524 posts from r/politics (1/1/20 - 5/1/20) with headline containing "Trump"
Trump_sentiment.csv - Flair, TextBlob, and VADER sentiment scores calculated from Trump_data.csv
Biden_data.csv - 11,138 posts from r/politics (1/1/20 - 5/1/20) with headline containing "Biden"
Trump_sentiment.csv - Flair, TextBlob, and VADER sentiment scores calculated from Biden_data.csv
