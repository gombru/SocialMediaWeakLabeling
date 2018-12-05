# Processes the tweets text file and creates a new text file with only tweets text and hashtags.
# Filters RT, short tweets, non-enflish and corrupted tweets.


import json
import urllib
import os.path

# -- CONFIG --
tweets_data_path = 'hate_tweets_1.txt'
min_text_length = 4
text_dir = '../../../datasets/HateSPic/twitter/txt/'
text_filename = 'hate_tweets_1.txt'

discard = ['naked','beautiful','sexy','videos','sex','nude','girl','porn','pics']


with open(text_dir + text_filename, "w") as text_file:
    processed = 0
    correct = 0
    # -- LOAD DATA -- each tweet is a dictionary
    tweets_file = open(tweets_data_path, "r")
    for line in tweets_file:
        try:
            t = json.loads(line)
        except:
            continue

        processed += 1
        if processed % 1000 == 0:
            print "Tweets processed: " + str(processed)
            print "Tweets accepted: " + str(correct)

        # Discard retweets
        if t.has_key('retweeted_status'): continue;

        # Discard non-enlish
        if not t.has_key('lang'): continue
        if t['lang'] != 'en': continue;

        # -- FILTER BY TEXT AND SAVE TEXT CONTENT -- discard short tweets
        if t.has_key(u'id') and t.has_key(u'text'):

            if len(t['text']) < min_text_length:
                print "Text too short: " + t['text']
                continue

            # hashtags_str = ''
            # if t.has_key(u'entities'):
            #     for hashtag in t['entities']['hashtags']:
            #         hashtags_str = hashtags_str + ',' + hashtag['text']

            # text = t['text'].encode("utf8", "ignore").replace('\n', ' ').replace('\r', '').replace(','," ") + ' ' + hashtags_str[1:].encode("utf8", "ignore").replace(',',' ')

            # Discard if containing words
            todiscard = False
            for w in discard:
                if text.__contains__(w):
                    # print "Discarding: " + text
                    todiscard = True
                    break
            if todiscard: continue
            # Save single text file with lines (id,text)
            text = t['text'].encode("utf8", "ignore").replace('\n', ' ').replace('\r', '')
            correct += 1
            text_file.write(str(t['id']) + ',' + text + '\n')



