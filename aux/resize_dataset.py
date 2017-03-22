# Resizes the images in a folder and creates a resized sets in another

import glob
from PIL import Image

images_path = "../../../../datasets/SocialMedia/img/trump"

c = 0
t = 0
for file in glob.glob(images_path + "/*.jpg"):
    t+=1
    print t
    try:
        im = Image.open(file).resize((300, 300), Image.ANTIALIAS)
        im.save(file.replace("img","imgResized"))
    except:
        c+=1
        # os.remove(file)
        # os.remove(file.replace("img","weak_ann").replace("jpg","txt"))
        print "Failed: " + str(c) + "   From: " + str(t)

