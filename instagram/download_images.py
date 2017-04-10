import os
import hues
import warnings
import time
import datetime
import dateutil.relativedelta
from core import InstaLooter
from utils import (warn_with_hues, warn_windows)
import myglobals

directory = '/home/imatge/disk2/cities_instagram/'
get_videos = False
login = 'pura_nata_djset'
password = 'Girasoles20'
jobs = 64
num_2_query= 200000 #Num of images to build the looter
num_2_download = 100000 #Num of images we want to end up having
new_only = False # Download only images newer than the current images in folder
myglobals.init()
myglobals.start_page = 0

cities = ['losangeles','chicago','melbourne','miami','sanfrancisco','sydney','toronto','singapore','london','newyork']


# warnings._showwarning = warnings.showwarning
# warnings.showwarning = warn_with_hues if os.name == "posix" else warn_windows

# Download posts from a certain period of time
# today = datetime.date.today()
# starting_month= 3
# month = starting_month
# timeframe = today - dateutil.relativedelta.relativedelta(month=starting_month - 1), today - dateutil.relativedelta.relativedelta(month=starting_month)
timeframe = None

first = True

while True:

    print "Sleeping"
    if not first: time.sleep(60*20)
    first = False
    c = 0

    while c < range(0, len(cities) - 1):

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
            continue

        try:
            # print "Starting page: " + str(myglobals.start_page )
            looter.download(media_count=num_2_query, new_only = new_only, timeframe = timeframe, with_pbar=False)
        except:
            print "Error while downloading, continuing ... "

        c +=1 # Go for next city

        # #Count number of images for curr city, and continue downloading if is not completed
        # if len(os.listdir(city_dir)) < num_2_download:
        #     print "Number of images for " + city + ': ' + str(len(os.listdir(city_dir)))
        #     print "Continue downloading images for same city..."
        #     myglobals.start_page = myglobals.start_page + 2
        #     # #Now get images of previous month
        #     # month += 1
        #     # timeframe = today- dateutil.relativedelta.relativedelta(months=month - 1), today - dateutil.relativedelta.relativedelta(months=month)
        #
        # else: #Go for the next city only if we have all the images for this city
        #     print "Number of images for " + city + ': ' + str(len(os.listdir(city_dir)))
        #     c+=1
        #     print c
        #     myglobals.start_page = 0
            #Start from 1st month
            # month = starting_month
            # timeframe = today - dateutil.relativedelta.relativedelta(month=starting_month - 1), today - dateutil.relativedelta.relativedelta(month=starting_month)

