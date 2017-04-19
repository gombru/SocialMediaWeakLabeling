# Copies images listed in a txt to a separate folder

from shutil import copyfile

file = "../../../datasets/SocialMedia/londonTestImages.txt"
img_path = "../../../datasets/SocialMedia/img_resized/cities_instagram/"
destPath = "../../../datasets/SocialMedia/samples/"

# num_images = 900
# n=0

with open(file, 'r') as fin:
    lines = fin.readlines()
    for line in lines:

        # n+=1
        # print n

        info = line.split(',')
        id = info[0]
        path = img_path + id + '.jpg'
        copyfile(path, destPath + id + '.jpg')

        # if n == num_images:
        #     break
