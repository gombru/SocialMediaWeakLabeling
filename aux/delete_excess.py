# Resizes the images in a folder and creates a resized datasetcd in another
# It also filters  corrupted images

import glob
import os
from shutil import copyfile

im_path = "/home/imatge/datasets/SocialMedia/img_resized_1M/cities_instagram/"
cap_path = "/home/imatge/datasets/SocialMedia/captions_resized_1M/cities_instagram/"
# cities = ['london','newyork','sydney','losangeles','chicago','melbourne','miami','toronto','singapore','sanfrancisco']
# cities = ['london', 'newyork', 'losangeles','melbourne','sydney','toronto']
cities = ['chicago','miami','sanfrancisco']

desired_num = 100000

for city in cities:

    print city

    for file in glob.glob(im_path + city + "/*.jpg"):

        # Check num of resized images. If we reach the desired number, stop
        if len(os.listdir(im_path + file.split('/')[-2] + '/')) <= desired_num: break
        os.remove(im_path + file.split('/')[-2] + '/' + file.split('/')[-1])
        os.remove(cap_path + file.split('/')[-2] + '/' + file.split('/')[-1].replace('jpg','txt'))






