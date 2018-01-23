# Embeddings
input = "/home/raulgomez/datasets/SocialMedia/regression_output/SocialMedia_Inception_frozen_word2vec_tfidfweighted_divbymax_iter_150000/test.txt"

out_embeddings = "/home/raulgomez/projects/TSNE/aux_data/SocialMedia_word2vec_onlyFood_embeddings.txt"
out_filelist = "/home/raulgomez/projects/TSNE/aux_data/SocialMedia_word2vec_onlyFood_filelist.txt"

search_words = ['food','eat','drink','dinner','breakfast','lunch','wine','beer','cocktail','restaurant','bar','cook','meat','fish','hamburguer','sushi','salad','fruit','vegetables','pasta','pizza']

c = 0

out_em_file = open(out_embeddings, 'w')
out_fl_file = open(out_filelist, 'w')


with open(input) as infile:
    for line in infile:
        c += 1
        if c % 10000 == 0: print c
        id = line.split(',')[0]
        # Get associated text
        use = False
        caption = ''
        with open('/home/raulgomez/datasets/SocialMedia/captions_resized_1M/cities_instagram/' + id + '.txt', 'r') as textfile:
            for text_line in textfile:
                caption = caption + text_line
        words = caption.split(' ')
        for w in words:
            if w in search_words:
                use = True
                break
        if use:
            out_fl_file.write(id + '.jpg\n')
            out_em_file.write(line)

print "DONE"


