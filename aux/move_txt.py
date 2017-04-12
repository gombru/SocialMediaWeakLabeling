# Moves all txt files in a folder to another folder

import glob
from shutil import copyfile
import os
from joblib import Parallel, delayed

path = "/home/imatge/disk2/cities_instagram/"
dest_path = "/home/imatge/disk2/cities_instagram/captions/"
cities = ['london','newyork','sydney','losangeles','chicago','melbourne','miami','toronto','singapore','sanfrancisco']


def move_txt(file):
    copyfile(file, file.replace('/cities_instagram/', '/cities_instagram/captions/'))
    os.remove(file)



for city in cities:
    if not os.path.exists(dest_path + city):
        os.makedirs(dest_path + city)
    print city
    Parallel(n_jobs=10)(delayed(move_txt)(file) for file in glob.glob(path + city + "/*.txt"))

print "END"