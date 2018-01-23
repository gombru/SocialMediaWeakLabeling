
# Embeddings
words = "/home/raulgomez/projects/TSNE/word_embeddings/SocialMedia_lda_500.txt"
img =  "/home/raulgomez/datasets/SocialMedia/regression_output/instagram_cities_1M_Inception_frozen_500_chunck_multiGPU_iter_500000/test.txt"
out = "/home/raulgomez/projects/TSNE/aux_data/SocialMedia_lda_500_embeddings.txt"
filenames = [words, img]
with open(out, 'w') as outfile:
    for fname in filenames:
        with open(fname) as infile:
            for line in infile:
                outfile.write(line)


# #File lists
# words = "/home/raulgomez/projects/TSNE/word_images/indices.txt"
# img =  "/home/raulgomez/projects/TSNE/SM_filelists/SocialMedia_lda_200_filelist.txt"
# out = "/home/raulgomez/projects/TSNE/aux_data/SocialMedia_lda_200_filelist.txt"
# filenames = [words, img]
# with open(out, 'w') as outfile:
#     for fname in filenames:
#         with open(fname) as infile:
#             for line in infile:
#                 outfile.write(line)
#
# print "DONE"