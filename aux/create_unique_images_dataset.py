# Takes a folder of images and finds duplicates using the im size and the file size. It creates a new folder with non-repeated images and another folder with the weak annotations of the
# non-repeated images. Then the LDA test should be run in that weak annotations file to create the splits to train the net

import glob
import os
import matplotlib.pyplot as plt
from PIL import Image
from shutil import copyfile

images_path = "/home/imatge/datasets/SocialMedia/img/trump"

i = 0
images = {}
paths = {}
for file in glob.glob(images_path + "/*.jpg"):
    i+=1
    s = os.path.getsize(file)
    im = Image.open(file)
    key = str(im.size[0]) + str(im.size[1]) + str(s)
    if not images.has_key(key):
        images[key] = 1

        copyfile(file.replace('img', 'imgResized'), file.replace('img', 'imgUnique'))
        copyfile(file.replace('img', 'weak_ann').replace('.jpg','.txt'), file.replace('img', 'weak_ann').replace('trump','trumpUnique').replace('.jpg','.txt'))

    else:
        images[key]+=1
    print i

values = sorted(images.values(), reverse = True)

non_unique = sum(i > 1 for i in values)
unique = sum(i == 1 for i in values)
doubled = sum(i == 2 for i in values)
triple = sum(i == 3 for i in values)


print "Total different: " + str(len(values))
print "Non Unique: " + str(non_unique)
print "Unique: " + str(unique)
print "Double: " + str(doubled)
print "Triple: " + str(triple)
print "More repeated: " + str(values[0])





