# Removes the corrupted image nfiles and its anotations. Then LDA test should be run to do the splits for training
# DEPRECATED: Now process_tweets_images.py is already filtering corrupted images

import glob
from PIL import Image
import os
from joblib import Parallel, delayed


images_path = "../../../../datasets/SocialMedia/img/trump"

c = 0
t = 0
threads = 10


def checkCorruption(file,t,c):
    t+=1
    try:
        im = Image.open(file)
    except:
        c+=1
        os.remove(file)
        os.remove(file.replace("img","tweets_info").replace("jpg","txt"))
        print "Removed: " + str(c) + "   From: " + str(t)


Parallel(n_jobs=threads)(delayed(checkCorruption)(file,t,c) for file in glob.glob(images_path + "/*.jpg"))




