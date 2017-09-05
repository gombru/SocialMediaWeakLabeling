# Creates train/val/test splits copying the files in separate folders

import glob
from shutil import copyfile
import os
import random

path = "/home/Imatge/datasets/SocialMedia/captions_resized_1M/cities_instagram/"
dest_path_captions = "/home/Imatge/datasets/InstaCities1M/captions/"
dest_path_img = "/home/Imatge/datasets/InstaCities1M/img/"
cities = ['london','newyork','sydney','losangeles','chicago','melbourne','miami','toronto','singapore','sanfrancisco']

train_th = 1e6 * 0.8
val_th = train_th + 1e6 * 0.05

def move_txt(file,split):
    copyfile(file, file.replace('/SocialMedia/captions_resized_1M/cities_instagram/', '/InstaCities1M/captions/' + split + '/'))
    os.remove(file)

def move_img(file,split):
    copyfile(file.replace('/captions_resized_1M/','/img_resized_1M/').replace('txt','jpg'), file.replace('/SocialMedia/captions_resized_1M/cities_instagram/', '/InstaCities1M/img/' + split + '/'))
    os.remove(file)



for city in cities:
    splits = ['train/','val/','test/']
    for s in splits:
        #Create dirs
        if not os.path.exists(dest_path_captions + s + city):
            os.makedirs(dest_path_captions + s + city)
        if not os.path.exists(dest_path_img + s + city):
            os.makedirs(dest_path_img + city)
    print city

    names = []
    #Load filenames
    for file in glob.glob(path + city + "/*.txt"):
        names.append(file)

    #Randomize names array
    names = random.huffle(names)

    #Copy captiond and image file in the split
    for i,name in enumerate(names):
        if i < train_th:
            move_txt(name, 'train')
            move_img(name, 'train')
        elif i < val_th:
            move_txt(name, 'val')
            move_img(name, 'val')
        else:
            move_txt(name, 'test')
            move_img(name, 'test')


print "END"