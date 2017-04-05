import os
import hues
import warnings
import time

from core import InstaLooter
from utils import (warn_with_hues, warn_windows)

directory = '../../../datasets/SocialMedia/img/cities_instagram/'
get_videos = False
login = 'raulgombru'
password = 'Girasoles19'
jobs = 32
num_2_query= 10000000 #Num of images to build the looter
num_2_download = 100000 #Num of images we want to end up having
new_only = False # Download only images newer than the current images in folder

cities = ['singapore','toronto','london','newyork','sydney','losangeles','chicago','melbourne','miami','sanfrancisco']


# warnings._showwarning = warnings.showwarning
# warnings.showwarning = warn_with_hues if os.name == "posix" else warn_windows

c = 0

while c < range(0,len(cities)):

    city = cities[c]
    city_dir = directory + city + '/'

    if not os.path.exists(city_dir):
        os.makedirs(city_dir)
    if not os.path.exists(city_dir.replace('img','captions')):
        os.makedirs(city_dir.replace('img','captions'))

    remaining_num_2_download = num_2_download - len(os.listdir(city_dir))
    if remaining_num_2_download < 0: remaining_num_2_download = 0

    print city
    print "Number of images for " + city + ': ' + str(len(os.listdir(city_dir))) + "  --  Remaining: " + str(remaining_num_2_download)

    if remaining_num_2_download < 1:
        c+=1
        continue


    time.sleep(1)

    looter = InstaLooter(directory=city_dir, hashtag=city,
        add_metadata=False, get_videos=get_videos, jobs=jobs)

    try:
        looter.login(login, password)
        hues.success('Logged in.')
    except:
        print "Error while loggining"
        time.sleep(2)

    try:
        looter.download(media_count=num_2_query, new_only = new_only, with_pbar=False)
    except:
        print "Error while downloading, continuing ... "

    #Count number of images for curr city, and continue downloading if is not completed
    if len(os.listdir(city_dir)) < num_2_download:
        print "Number of images for " + city + ': ' + str(len(os.listdir(city_dir)))
        print "Continue downloading images for same city..."
    else: #Go for the next city only if we have all the images for this city
        print "Number of images for " + city + ': ' + str(len(os.listdir(city_dir)))
        c+=1
        print c
