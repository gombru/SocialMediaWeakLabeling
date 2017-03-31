# Resizes the images in a folder and creates a resized sets in another

import glob
from PIL import Image
from joblib import Parallel, delayed
import os

images_path = "../../../datasets/SocialMedia/img_unique/cities_1day/"
cities = ['paris','istanbul','rome','prague','milan','barcelona','amsterdam','vienna','moscow','berlin','madrid']

# c = 0
# t = 0

def resize(file):
    # print t
    saved = False
    try:
        im = Image.open(file).resize((300, 300), Image.ANTIALIAS)
        im.save(file.replace("img_unique","imgResized"))
        saved = True
    except:
        # c+=1
        # os.remove(file)
        # os.remove(file.replace("img","weak_ann").replace("jpg","txt"))
        print "Failed: " #+ str(c) + "   From: " + str(t)

for city in cities:
    if not os.path.exists(images_path.replace("img_unique","imgResized") + '/' + city):
        os.makedirs(images_path.replace("img_unique","imgResized") + '/' + city)
    Parallel(n_jobs=10)(delayed(resize)(file) for file in glob.glob(images_path + city + "/*.jpg"))
