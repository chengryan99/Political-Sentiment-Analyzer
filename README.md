# Political Sentiment Analyzer
## University of Colorado Denver | CSCI4930 - Final Project | Ryan Cheng<br/>

### Project Overview:<br/>
BasicSA.py - basic test using Vader SA on 1000 headlines, view negative/positive word distribution and tokenizers/stopwords<br/>
RetrieveData.py - retrieves data from Reddit using Pushshift API<br/>
SentimentAnalysis.py - performs sentiment analysis on data using Flair, Vader, and TextBlob models, outputs results from each model<br/>
PlotResults.py - plots results to show trend in sentiment over time<br/>
<br/>
reddit_headlines_labels.csv - sentiment labels using VADER from BasicSA.py<br/>
Trump_data.csv - 35,524 posts from r/politics (1/1/20 - 5/1/20) with headline containing "Trump"<br/>
Trump_sentiment.csv - Flair, TextBlob, and VADER sentiment scores calculated from Trump_data.csv<br/>
Biden_data.csv - 11,138 posts from r/politics (1/1/20 - 5/1/20) with headline containing "Biden"<br/>
Trump_sentiment.csv - Flair, TextBlob, and VADER sentiment scores calculated from Biden_data.csv<br/>
