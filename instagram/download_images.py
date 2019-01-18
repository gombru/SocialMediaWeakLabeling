import os
import time
import requests.packages.urllib3
import random

print("RUN WITH PYTHON 3!!")

requests.packages.urllib3.disable_warnings()
directory = '../../../hd/datasets/insta10YearsChallenge/data/'
# queries_file = '../instagram/queries_xmas.txt'
queries = []
# for line in open(queries_file,'r'):
#     queries.append(line.strip('\n'))
queries.append("10yearschallenge")
get_videos = False

num_2_query_first = 1000000 #Num of images to build the looter the first time (will downlaod old images)
num_2_query_next = 100000 #Num of images to build the looter affter (will download new updated images)
num_2_download = 2000000 #Num of images we want to end up having
sleep_seconds = 60 #120*60
first = True
num_2_query = num_2_query_first

while True:

    if not first:
        num_2_query = num_2_query_next
        print "Sleeping for: " + str(sleep_seconds / 60) + " minutes"
        time.sleep(sleep_seconds)
    first = False
    c = 0

    # for c in range(0, len(queries)):
    query_index = random.randint(0,len(queries)-1)
    query = queries[query_index]
    query_dir = directory #+ query + '/'
    print("Downloading images for: " + query)

    if not os.path.exists(query_dir):
        os.makedirs(query_dir)

    remaining_num_2_download = num_2_download - len(os.listdir(query_dir))
    if remaining_num_2_download < 0: remaining_num_2_download = 0

    print "Number of images for " + query + ': ' + str(len(os.listdir(query_dir))) + "  --  Remaining: " + str(remaining_num_2_download)

    if remaining_num_2_download < 1:
        continue

    try:
        os.system("instalooter hashtag " + query + " " + directory + " -n " + str(num_2_query) + " -j 32  -d " )

    except:
        print("Error on lotter, continuing")
