# Resizes the images in a folder and creates a resized datasetcd in another
# It also filters  corrupted images

import glob
from PIL import Image
from joblib import Parallel, delayed
import os
from shutil import copyfile

images_path = "/home/imatge/disk2/cities_instagram/img/"
im_dest_path = "/home/imatge/datasets/SocialMedia/img_resized_1M/cities_instagram/"
cap_dest_path = "/home/imatge/datasets/SocialMedia/captions_resized_1M/cities_instagram/"
aux_captions_path =  "/home/imatge/datasets/SocialMedia/captions_resized/cities_instagram/"
# cities = ['london','newyork','sydney','losangeles','chicago','melbourne','miami','toronto','singapore','sanfrancisco']
# cities = ['london', 'newyork', 'losangeles','melbourne','sydney','toronto']
# cities = ['chicago','miami','sanfrancisco']
cities = ['singapore']
desired_num = 100000

# c = 0
# t = 0

def resize(file):

    # Check num of resized images. If we reach the desired number, stop
    if len(os.listdir(im_dest_path + file.split('/')[-2] + '/')) >= desired_num: return
    # print len(os.listdir(im_dest_path + file.split('/')[-2] + '/'))
    #Check if resized file already exists
    if os.path.exists( cap_dest_path + file.split('/')[-1].replace('.jpg', '.txt')): return

    try:
        im = Image.open(file).resize((300, 300), Image.ANTIALIAS)
        im.save(im_dest_path + file.split('/')[-2] + '/' + file.split('/')[-1])
    except:
        print "Failed copying image: " #+ str(c) + "   From: " + str(t)
        return


    try:
        copyfile(file.replace('img', 'captions').replace('.jpg', '.txt'), cap_dest_path + file.split('/')[-2] + '/' + file.split('/')[-1].replace('.jpg', '.txt'))
        return
    except:
        print "Failed copying text from original"


    try:
        copyfile(aux_captions_path + file.split('/')[-2] + '/' + file.split('/')[-1].replace('.jpg', '.txt'), cap_dest_path + file.split('/')[-2] + '/' + file.split('/')[-1].replace('.jpg', '.txt'))
        print "Text copied from aux"
        return
    except:
        print "Failed copying text from aux"
        os.remove(im_dest_path + file.split('/')[-2] + '/' + file.split('/')[-1])



for city in cities:
    print city
    if not os.path.exists(im_dest_path + city):
        os.makedirs(im_dest_path + city)
    if not os.path.exists(cap_dest_path + city):
        os.makedirs(cap_dest_path + city)
    Parallel(n_jobs=4)(delayed(resize)(file) for file in glob.glob(images_path + city + "/*.jpg"))
