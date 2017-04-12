# Downloads instagram images and captions given query words.
# It's web-based, it does a search and goes page by page retrivieng results
# Once in page 3XXX the results start to be repeated. So this script checks if the file to be downloaded already exists,
# It stops downloading when N already existing images have been tried to download.
# It fires 4 ever every X minutes, so new uploaded images have been uploaded and can be dowloaded as the first results.
# Tried to do a controled navigation of page results, but did not work, still repeated results.


import os
import hues
import time
from core import InstaLooter
import requests.packages.urllib3
requests.packages.urllib3.disable_warnings()

directory = '/home/imatge/disk2/cities_instagram/img/'
get_videos = False
first = True
login = 'lazarpitas'
password = 'Gata20'
new_only = False # Download only images newer than the current images in folder

cities = ['losangeles','chicago','melbourne','miami','sanfrancisco','sydney','toronto','singapore','london','newyork']

jobs = 16
num_2_query= 200000 #Num of images to build the looter
num_2_download = 100000 #Num of images we want to end up having

sleep_seconds = 60*30


while True:

   if not first:
       print "Sleeping for: " + str(sleep_seconds / 60) + " minutes"
       time.sleep(sleep_seconds)
   first = False
   c = 0

   for c in range(0, len(cities)):

        city = cities[c]
        city_dir = directory + city + '/'

        if not os.path.exists(city_dir):
            os.makedirs(city_dir)
        if not os.path.exists(city_dir.replace('img','captions')):
            os.makedirs(city_dir.replace('img','captions'))

        remaining_num_2_download = num_2_download - len(os.listdir(city_dir))
        if remaining_num_2_download < 0: remaining_num_2_download = 0

        print "Number of images for " + city + ': ' + str(len(os.listdir(city_dir))) + "  --  Remaining: " + str(remaining_num_2_download)

        if remaining_num_2_download < 1:
            continue

        looter = InstaLooter(directory=city_dir, hashtag=city,
            add_metadata=False, get_videos=get_videos, jobs=jobs)

        try:
            looter.login(login, password)
            hues.success('Logged in.')
        except:
            print "Error while loggining"
            time.sleep(60)
            continue

        try:
            looter.download(media_count=num_2_query, new_only = new_only, with_pbar=False)
        except:
            print "Error while downloading, continuing ... "