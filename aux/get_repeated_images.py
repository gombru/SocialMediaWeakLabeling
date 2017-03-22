# Counts repeated image files by im size and file size and plots it

import glob
import os
import matplotlib.pyplot as plt
from PIL import Image

images_path = "/home/imatge/datasets/SocialMedia/img/trump"

i = 0
images = {}
for file in glob.glob(images_path + "/*.jpg"):
    i+=1
    s = os.path.getsize(file)
    im = Image.open(file)
    key = str(im.size[0]) + str(im.size[1]) + str(s)
    if not images.has_key(key):
        images[key] = 1
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

# -- Plot histogram
# plt.hist(values[:50])
# plt.title("Gaussian Histogram")
# plt.xlabel("Value")
# plt.ylabel("Frequency")
# fig = plt.gcf()
# plt.show()

# -- Plot plain
plt.bar(range(len(values[:non_unique])), values[:non_unique], align='center')
plt.ylabel("Times repeated")
plt.axis((0,non_unique,0,200))
plt.show()




