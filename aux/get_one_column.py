# Embeddings
input = "/home/raulgomez/datasets/instaFashion/regression_output/instaFashion_Inception_frozen_word2vec_tfidf_iter_75000/test.txt"
out = "/home/raulgomez/projects/TSNE/instaFashion/instaFashion_Inception_frozen_word2vec_tfidf_iter_75000_filelist.txt"
with open(out, 'w') as outfile:
    with open(input) as infile:
        for line in infile:
            txt = line.split(',')[0] + '.jpg\n'
            outfile.write(txt)
outfile.close()
