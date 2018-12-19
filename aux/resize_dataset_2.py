
import glob
from PIL import Image
from joblib import Parallel, delayed
import os


json_path = "../../../hd/datasets/instaMiro/data/"
images_path = "../../../hd/datasets/instaMiro/data/"
im_dest_path = "../../../hd/datasets/instaMiro/img_resized/"

minSize = 300


def resize(file):
    try:
        im_path = images_path + file.replace("json", "jpg").split('/')[-1]
        im = Image.open(im_path)

        w = im.size[0]
        h = im.size[1]

        # print "Original w " + str(w)
        # print "Original h " + str(h)

        if w < h:
            new_width = minSize
            new_height = int(minSize * (float(h) / w))

        if h <= w:
            new_height = minSize
            new_width = int(minSize * (float(w) / h))

        # print "New width "+str(new_width)
        # print "New height "+str(new_height)
        im = im.resize((new_width, new_height), Image.ANTIALIAS)
        im.save(im_dest_path + im_path.split('/')[-1])

    except:
        print "Failed copying image" #. Removing image and caption"
        # try:
        #     print "Removing"
        #     #os.remove(im_path)
        #     #os.remove(file)
        # except:
        #     print "Cannot remove"
        #     return
        # print "Removed"
        return


if not os.path.exists(im_dest_path):
    os.makedirs(im_dest_path)
Parallel(n_jobs=12)(delayed(resize)(file) for file in glob.glob(json_path + "/*.json"))