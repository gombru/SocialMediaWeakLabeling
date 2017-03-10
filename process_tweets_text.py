import json
import urllib
import os.path

# -- CONFIG --
tweets_data_path = '../../datasets/SocialMedia/data/tweets_data_trump_10-3-17.txt'
min_text_length = 10
text_dir = '../../datasets/SocialMedia/text/'
text_filename = 'text_trump_test_big.txt'


# -- LOAD DATA -- each tweet is a dictionary
tweets = []
tweets_file = open(tweets_data_path, "r")
for line in tweets_file:
    try:
        t = json.loads(line)
        tweets.append(t)
    except:
        continue

print "Number of tweets to process: " + str(len(tweets))

c = 0

with open(text_dir + text_filename, "w") as text_file:

    for t in tweets:

        c += 1
        if c % 100 == 0:
            print c

        # -- FILTER BY TEXT AND SAVE TEXT CONTENT -- discard short tweets
        if t.has_key(u'id') and t.has_key(u'text') and t.has_key(u'created_at'):

            if len(t['text']) < min_text_length:
                print "Text too short: " + t['text']
                continue

            hashtags_str = ''
            if t.has_key(u'entities'):
                for hashtag in t['entities']['hashtags']:
                    hashtags_str = hashtags_str + ',' + hashtag['text']

            # Save single text file with lines (id,text)
            text_file.write(
                str(t['id']) + ',' + t['text'].encode(
                    "utf8", "ignore").replace('\n', ' ').replace('\r', '').replace(','," ") + ' ' + hashtags_str[1:].encode("utf8", "ignore").replace(',',' ') + '\n')



