# Resizes the images in a folder and creates a resized datasetcd in another
# It also filters  corrupted images

import glob
from PIL import Image
from joblib import Parallel, delayed
import os

images_path = "/home/imatge/datasets/SocialMedia/img/cities_instagram/"
cities = ['london','newyork','sydney','losangeles','chicago','melbourne','miami','toronto','singapore','sanfrancisco']

# c = 0
# t = 0

def resize(file):
    # print t
    # saved = False
    try:
        im = Image.open(file).resize((300, 300), Image.ANTIALIAS)
        im.save(file.replace("img","img_resized"))
        # saved = True
    except:
        # c+=1
        # os.remove(file)
        # os.remove(file.replace("img","weak_ann").replace("jpg","txt"))
        print "Failed: " #+ str(c) + "   From: " + str(t)


for city in cities:
    if not os.path.exists(images_path.replace("img","img_resized") + '/' + city):
        os.makedirs(images_path.replace("img","img_resized") + '/' + city)
    Parallel(n_jobs=10)(delayed(resize)(file) for file in glob.glob(images_path + city + "/*.jpg"))
