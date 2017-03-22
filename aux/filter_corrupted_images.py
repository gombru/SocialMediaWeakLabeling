# Removes the corrupted image nfiles and its anotations. Then LDA test should be run to do the splits for training

import glob
from PIL import Image
import os

images_path = "../../../../datasets/SocialMedia/img/trump"

c = 0
t = 0
for file in glob.glob(images_path + "/*.jpg"):
    t+=1

    try:
        im = Image.open(file)
    except:
        c+=1
        os.remove(file)
        os.remove(file.replace("img","weak_ann").replace("jpg","txt"))
        print "Removed: " + str(c) + "   From: " + str(t)

