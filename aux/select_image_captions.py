import glob
import os
import matplotlib.pyplot as plt
from PIL import Image
from shutil import copyfile


images_path = "/home/imatge/datasets/SocialMedia/img_resized/cities_instagram/"
cities = ['london','newyork','sydney','losangeles','chicago','melbourne','miami','toronto','singapore','sanfrancisco']

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
            copyfile(file.replace('img_resized', 'captions').replace('.jpg','.txt'), file.replace('img_resized', 'captions_resized').replace('.jpg','.txt'))
        except:
            print "Caption file does not exist: " + file.replace('img_resized', 'captions').replace('.jpg','.txt')
            print "TO: " + file.replace('img_resized', 'captions_resized').replace('.jpg','.txt')

print "END"
