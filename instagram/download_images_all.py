# Downloads instagram images and captions given query words.
# It's web-based, it does a search and goes page by page retrivieng results
# Once in page 3XXX the results start to be repeated. So this script checks if the file to be downloaded already exists,
# It stops downloading when N already existihuesng images have been tried to download.
# It fires 4 ever every X minutes, so new uploaded images have been uploaded and can be dowloaded as the first results.
# Tried to do a controled navigation of page results, but did not work, still repeated results.


import os
import hues
import time
from core import InstaLooter
import requests.packages.urllib3
requests.packages.urllib3.disable_warnings()

directory = '/home/Imatge/hd/datasets/instaFashion/img/'
# directory = '/home/raulgomez/datasets/instaFashion/img/'
get_videos = False
first = True
login1 = login = 'bruballa_gomez'
password1 = password ='Gata20'
login2 = 'bruballa_gomez'
password2 = 'Gata20'
new_only = False # Download only images newer than the current images in folder

queries = open('queries_fashion.txt')
jobs = 16
num_2_query_first = 50000 #2500000 #Num of images to build the looter the first time (will downlaod old images)
num_2_query_next = 50000 #50000 #Num of images to build the looter affter (will download new updated images)
num_2_download = 50000 #5000000 #Num of images we want to end up having

sleep_seconds = 60*30


num_2_query = num_2_query_first

while True:

    if not first:
        num_2_query = num_2_query_next
        print "Sleeping for: " + str(sleep_seconds / 60) + " minutes"
        time.sleep(sleep_seconds)
    first = False
    c = 0

    for query in queries:

        q = query.split('#')[-1].strip('\n')
        dir = directory + q + '/'

        if not os.path.exists(dir):
            os.makedirs(dir)
        if not os.path.exists(dir.replace('img','json')):
            os.makedirs(dir.replace('img','json'))

            remaining_num_2_download = num_2_download - len(os.listdir(dir))
            if remaining_num_2_download < 0: remaining_num_2_download = 0

            print "Number of images for " + q + ': ' + str(len(os.listdir(dir))) + "  --  Remaining: " + str(remaining_num_2_download)

            if remaining_num_2_download < 1:
                continue

            looter = InstaLooter(directory=dir, hashtag=q,
                add_metadata=False, get_videos=get_videos, jobs=jobs)

            try:
                looter.login(login, password)
                hues.success('Logged in.')
            except:
                print "Error while loggining"
                if login == login1:
                    login = login2
                    password = password2
                else:
                    login = login1
                    password = password1

                time.sleep(60)
                continue

            try:
                looter.download(media_count=num_2_query, new_only = new_only, with_pbar=False)
            except:
                print "Error while downloading, continuing ... "

        print "Sleeping for: " + str(60*5 / 60) + " minutes"
        time.sleep(sleep_seconds)
        q = query.split('#')[-1].strip('\n')
        dir = directory + q + '/'