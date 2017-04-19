# Copies to a new folder the txt files with captions associated to images in a folder.
# DEPRECATED: resize_dataset already copies captions

import glob
import os
from shutil import copyfile


images_path = "/home/imatge/datasets/SocialMedia/img_resized_1M/cities_instagram/"
# cities = ['london','newyork','sydney','losangeles','chicago','melbourne','miami','toronto','singapore','sanfrancisco']
cities = ['london']

i = 0
images = {}
paths = {}

for city in cities:

    if not os.path.exists(images_path.replace('img_resized', 'captions_resized') + '/' + city):
        os.makedirs(images_path.replace('img_resized', 'captions_resized') + '/' + city)

for c in cities:
    for file in glob.glob(images_path + c + "/*.jpg"):
        i+=1
        try:
            copyfile(file.replace('img_resized', 'captions_resized').replace('.jpg','.txt'), file.replace('img_resized', 'captions_resized').replace('.jpg','.txt').replace('london','london2'))
        except:
            print "Caption file does not exist: " + file.replace('img_resized', 'captions').replace('.jpg','.txt')
            print "TO: " + file.replace('img_resized', 'captions_resized').replace('.jpg','.txt')

print "END"
