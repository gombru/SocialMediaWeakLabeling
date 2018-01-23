import os

f = open('indices.txt','w')
for file in os.listdir('../../../projects/TSNE/word_images/'):
    f.write('word_images' + '/' + file + '\n')
    print file