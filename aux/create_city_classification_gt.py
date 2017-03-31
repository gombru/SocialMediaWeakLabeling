# Creates city classification gt files for caffe CNN

import glob
from random import randint

images_path = "/home/imatge/datasets/SocialMedia/img/cities_instagram/"

cities = ['london','newyork','sydney','losangeles','chicago','melbourne','miami','toronto','singapore','sanfrancisco']

gt_path_train = '../../../datasets/SocialMedia/city_classification_gt/cities_instagram/trainCitiesClassification.txt'
gt_path_val = '../../../datasets/SocialMedia/city_classification_gt/cities_instagram/valCitiesClassification.txt'
gt_path_test = '../../../datasets/SocialMedia/city_classification_gt/cities_instagram/testCitiesClassification.txt'
train_file = open(gt_path_train, "w")
val_file = open(gt_path_val, "w")
test_file = open(gt_path_test, "w")


i = 0
images = {}

for c in range(0,len(cities)):
    city = cities[c]
    for file in glob.glob(images_path + city + "/*.jpg"):

        #Create splits
        split = randint(0, 9)
        if split < 8:
            train_file.write(city + '/' + file.split('/')[-1][:-4] + ',' + str(c) + '\n')
        elif split == 8:
            val_file.write(city + '/' + file.split('/')[-1][:-4] + ',' + str(c) + '\n')
        else:
            test_file.write(city + '/' + file.split('/')[-1][:-4] + ',' + str(c) + '\n')

train_file.close()
val_file.close()
test_file.close()