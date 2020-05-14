"""
Retrieves Reddit data for sentiment analysis, allows for more data than using Praw
Resource: https://medium.com/@RareLoot/using-pushshifts-api-to-extract-reddit-submissions-fb517b286563
"""

import numpy as np
import requests
import json
import csv
import time
import datetime
import os
import io

# function to set up data retrieval from pushshift API
def get_pushshift_data(query, after, before, sub):
    url = 'https://api.pushshift.io/reddit/search/submission/?title=' + str(query) + '&size=1000&after=' + str(
        after) + '&before=' + str(before) + '&subreddit=' + str(sub)
    print(url)
    r = requests.get(url)
    # load as json
    data = json.loads(r.text)
    return data['data']

# function to retrieve and format data
def collect_sub_data(subm):
    sub_data = list()  # list to store data points
    title = subm['title']
    url = subm['url']
    # check if flair is available
    try:
        flair = subm['link_flair_text']
    except KeyError:
        flair = 'NaN'
    author = subm['author']
    sub_id = subm['id']
    score = subm['score']
    try:
        # if selftext is available then get it, else set it empty
        selftext = subm['selftext']
        list_of_empty_markers = ['[removed]', '[deleted]']
        if selftext in list_of_empty_markers:
            selftext = ''
    except:
        selftext = ''
    created = datetime.datetime.fromtimestamp(subm['created_utc'])
    numComms = subm['num_comments']
    permalink = subm['permalink']

    sub_data.append((sub_id, title, selftext, url, author, score, created, numComms, permalink, flair))
    sub_stats[sub_id] = sub_data

# function to write data to csv file
def write_subs_to_file(filename):
    upload_count = 0
    if os.path.exists(filename):
        keep_header = False
    else:
        keep_header = True

    with io.open(filename, 'a', newline='', encoding="utf-8") as file:
        a = csv.writer(file, delimiter=',')
        headers = ['post_id', 'title', 'selftext', 'url', 'author', 'score', 'publish_date', 'num_of_comments',
                   'permalink', 'flair']
        if keep_header:
            a.writerow(headers)
        for sub in sub_stats:
            a.writerow(sub_stats[sub][0])
            upload_count += 1
        print(str(upload_count) + ' submissions have been uploaded')

if __name__ == '__main__':
    # specify subreddit and keyword
    sub_reddit = 'politics'
    key_word = 'Biden'

    output_filename = 'Biden_data.csv'
    # specify start and end date for posts
    start_date = datetime.datetime(2020, 1, 1, 0)
    end_date = datetime.datetime(2020, 5, 1, 0)

    # each iteration retrieves one day of posts
    one_day = datetime.timedelta(hours=24)
    after_date = start_date
    after = str(int(after_date.timestamp()))
    before_date = start_date + one_day
    before = str(int(before_date.timestamp()))

    while after_date < end_date:
        print('-' * 80)
        print(after_date, ' -> ', before_date)
        print('-' * 80)

        sub_count = 0
        sub_stats = {}

        data = get_pushshift_data(key_word, after, before, sub_reddit)

        max_count = 100
        count = 0
        while len(data) > 0 and count < max_count:
            print('count ', count)
            for submission in data:
                collect_sub_data(submission)
                sub_count += 1

            print(len(data))
            print(str(datetime.datetime.fromtimestamp(data[-1]['created_utc'])))
            after = data[-1]['created_utc']
            data = get_pushshift_data(key_word, after, before, sub_reddit)
            # print(data)
            # print(data['data'][0]['author'])
            count = count + 1

        # save data for each iteration
        write_subs_to_file(output_filename)

        # continue for next day in time period
        after_date += one_day
        after = str(int(after_date.timestamp()))
        before_date += one_day
        before = str(int(before_date.timestamp()))

        # delay between iterations to avoid being blocked from server
        time.sleep(np.random.randint(1, 3))