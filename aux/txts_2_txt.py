import glob

instagram_text_data_path = '../../../datasets/SocialMedia/captions_resized_1M/cities_instagram/'
output_path = '../../../datasets/SocialMedia/captions_resized_1M/cities_instagram/captions_1M_alltxt.txt'
output_file = open(output_path, "w")
cities = ['london','newyork','sydney','losangeles','chicago','melbourne','miami','toronto','singapore','sanfrancisco']


for city in cities:
    print "Loading data from " + city
    for file_name in glob.glob(instagram_text_data_path + city + "/*.txt"):
        caption = ""
        file = open(file_name, "r")
        for line in file:
            caption = caption + line

        output_file.write(caption.replace('\n', ' ') + '\n')
