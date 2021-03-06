# Saves images examples of each topic for visualization
# DEPRECATED: This was uses for twitter images

import os
from PIL import Image

file = "../../../datasets/SocialMedia/valCities1Day.txt"
img_path = "../../../datasets/SocialMedia/img_unique/cities_1day/"
destPath = "../../../datasets/SocialMedia/samples/cities_1day"
num_images = 900

n=0

with open(file, 'r') as fin:
    lines = fin.readlines()
    for line in lines:

        n+=1
        print n

        info = line.split(',')
        id = info[0]
        label = info[1][:-1]

        if not os.path.exists(destPath + str(label)):
            os.makedirs(destPath + str(label))

        im = Image.open(img_path + id + '.jpg')
        im.save(destPath + str(label) + '/' + str(id) + '.jpg')

        if n == num_images:
            break
