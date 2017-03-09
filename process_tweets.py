import json
import urllib

# -- CONFIG --
tweets_data_path = '../../datasets/SocialMedia/data/twitter_data.txt'
min_text_length = 8
images_dir = '../../datasets/SocialMedia/img/'
ann_dir = '../../datasets/SocialMedia/ann/'



def download_save_image(url, filename):
    resource = urllib.urlopen(url)
    output = open(images_dir + filename, "wb")
    output.write(resource.read())
    output.close()


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

# -- FILTER BY IMAGE AND SAVE IMAGES -- discard tweets without image
print "Filtering by image ..."
c = 0
for t in tweets:

    c += 1
    if c % 100 == 0:
        print c

    if t.has_key(u'entities'):
        if t['entities'].has_key(u'media'):
            if t['entities']['media'][0]['type'] == u'photo':
                download_save_image(t['entities']['media'][0]['media_url'], str(t['id']) + ".jpg")
                print t['entities']['media'][0]['media_url']
                continue

    tweets.remove(t)

# -- FILTER BY TEXT AND SAVE TEXT CONTENT -- discard short tweets
print "Filtering by text ..."
c = 0
for t in tweets:

    c += 1
    if c % 100 == 0:
        print c

    if t.has_key(u'text'):
        if len(t['text']) < min_text_length:
            print "Text too short: " + t['text']
            tweets.remove(t)
    else:
        tweets.remove(t)

    hashtags_str = ''
    if t.has_key(u'entities'):
        for hashtag in t['entities']['hashtags']:
            hashtags_str = hashtags_str + ',' + hashtag['text']

    print t['id']

    with open(images_dir + str(t['id']) + '.txt', "w") as text_file:
        text_file.write(str(t['id']) + '\n' + t['created_at'].encode("utf8", "ignore") + '\n' + t['text'].encode("utf8", "ignore") + '\n' + hashtags_str[1:].encode("utf8", "ignore"))


print "Number of remaining tweets after filtering: " + str(len(tweets))




