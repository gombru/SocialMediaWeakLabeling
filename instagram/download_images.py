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
add_metadata = True
jobs = 32
num_2_download = 20000

cities = ['london','newyork','sydney','losangeles','chicago','melbourne','miami','toronto','singapore','sanfrancisco']


warnings._showwarning = warnings.showwarning
warnings.showwarning = warn_with_hues if os.name == "posix" else warn_windows



for city in cities:
    city_dir = directory + city + '/'
    time.sleep(10)
    if not os.path.exists(city_dir):
        os.makedirs(city_dir)
    if not os.path.exists(city_dir.replace('img','captions')):
        os.makedirs(city_dir.replace('img','captions'))

    looter = InstaLooter(directory=city_dir, hashtag=city,
        add_metadata=False, get_videos=get_videos, jobs=jobs)

    looter.login(login, password)
    hues.success('Logged in.')

    looter.download(media_count=num_2_download, new_only = True, with_pbar=False)
