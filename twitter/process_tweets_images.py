# Processes the tweets text file and saves 2 files per tweets, both names with tweet id:
    # Image file
    # Annotation file: Tweet text, date and hashtags
# Filters already existing images (by id), RT, short tweets, non-english tweets and corrupted tweets.

import json
import urllib
import os.path
from joblib import Parallel, delayed
from PIL import Image


# -- CONFIG --
tweets_data_path = 'hate_tweets_1.txt'
min_text_length = 4
images_dir = '../../../datasets/HateSPic/twitter/img/'
tweets_info_dir = '../../../datasets/HateSPic/twitter/txt/'

discard = ['naked','beautiful','sexy','videos','sex','nude','girl','porn','pics']

threads = 10


def download_save_image(url, image_path):
    resource = urllib.urlopen(url)
    output = open(image_path, "wb")
    output.write(resource.read())
    output.close()


def process_tweet(line):
    # Discard short lines
    if len(line) < min_text_length: return

    try:
        t = json.loads(line)
    except:
        print "Failed to load tweet json, continuing"
        return

    # c += 1
    # if c % 1000 == 0:
    #     print "Num of tweets analyzed:" + str(c)
    #     print "Num of images downloaded:" + str(i)

    # Discard tweets without mandatory fields
    if not t.has_key(u'id'): return
    if not t.has_key(u'text'): return

    # Discard retweets
    if t.has_key('retweeted_status'): return;

    # Discard non-enlish
    if not t.has_key('lang'): return
    if t['lang'] != 'en': return;

    # Discard short tweets
    if len(t['text']) < min_text_length:
        print "Text too short: " + t['text']
        return

    # -- FILTER BY IMAGE AND SAVE IMAGES -- discard tweets without image
    if t.has_key(u'id') and t.has_key(u'entities'):

        #Check if tweet has jpg image
        if t['entities'].has_key(u'media'):
            if t['entities']['media'][0]['type'] == u'photo':
                if t['entities']['media'][0]['media_url'][-3:] != 'jpg':
                    return

                text = t['text'].encode("utf8", "ignore").replace('\n', ' ').replace('\r', '')

                # Discard if containing words
                for w in discard:
                    if text.__contains__(w):
                        # print "Discarding: " + text
                        return


                image_path = images_dir + '/' + str(t['id']) + ".jpg"
                # Check if file already exists
                if os.path.isfile(image_path):
                    print "Image already exists"
                    return

                # Download image
                try:
                    download_save_image(t['entities']['media'][0]['media_url'], image_path)
                    # Check image can be opened
                    im = Image.open(image_path)
                except:
                    if os.path.exists(image_path):
                        os.remove(image_path) #Remove the corrupted file
                    print "Failed downloading image from: " + t['entities']['media'][0]['media_url']
                    return


                with open(tweets_info_dir + '/' +str(t['id']) + '.txt', "w") as text_file:
                    text_file.write(str(t['id']) + text + '\n')



# -- LOAD DATA -- each tweet is a dictionary
tweets_file = open(tweets_data_path, "r")
Parallel(n_jobs=threads)(delayed(process_tweet)(line,) for line in tweets_file)




