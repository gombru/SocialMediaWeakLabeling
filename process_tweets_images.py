import json
import urllib
import os.path

# -- CONFIG --
tweets_data_path = '../../datasets/SocialMedia/data/twitter_data.txt'
min_text_length = 10
images_dir = '../../datasets/SocialMedia/img/'
ann_dir = '../../datasets/SocialMedia/ann/'


def download_save_image(url, filename):
    resource = urllib.urlopen(url)
    output = open(images_dir + filename, "wb")
    output.write(resource.read())
    output.close()


# -- LOAD DATA -- each tweet is a dictionary
tweets_file = open(tweets_data_path, "r")
c=0
for line in tweets_file:

    c += 1
    if c % 100 == 0:
        print c

    try:
        t = json.loads(line)
    except:
        continue

    # Check if file already exists
    if os.path.isfile(images_dir + str(t['id']) + ".jpg"): continue

    # -- FILTER BY IMAGE AND SAVE IMAGES -- discard tweets without image
    if t.has_key(u'id') and t.has_key(u'entities'):

        #Check if tweet has jpg image
        if t['entities'].has_key(u'media'):
            if t['entities']['media'][0]['type'] == u'photo':
                if t['entities']['media'][0]['media_url'][-3:] != 'jpg':
                    continue

                #Download image
                try:
                    download_save_image(t['entities']['media'][0]['media_url'], str(t['id']) + ".jpg")
                except:
                    print "Failed downloading image from: " + t['entities']['media'][0]['media_url']
                    continue

                print t['entities']['media'][0]['media_url']

                # -- FILTER BY TEXT AND SAVE TEXT CONTENT -- discard short tweets
                if t.has_key(u'id') and t.has_key(u'text') and t.has_key(u'created_at'):

                    if len(t['text']) < min_text_length:
                        print "Text too short: " + t['text']
                        continue

                    hashtags_str = ''
                    if t.has_key(u'entities'):
                        for hashtag in t['entities']['hashtags']:
                            hashtags_str = hashtags_str + ',' + hashtag['text']

                    with open(ann_dir + str(t['id']) + '.txt', "w") as text_file:
                        text_file.write(
                            str(t['id']) + '\n' + t['created_at'].encode("utf8", "ignore") + '\n' + t['text'].encode(
                                "utf8", "ignore").replace('\n', ' ').replace('\r', '') + '\n' + hashtags_str[1:].encode("utf8", "ignore"))



