import numpy as np
import random
import matplotlib.pyplot as plt
import glob
embedding = 'SocialMedia_Inception_frozen_word2vec_tfidfweighted_divbymax_iter_150000'
data_file = '../../../datasets/MIRFLICKR25K/both_embeddings/'+ embedding + '/data.txt'
filtered_topics = '../../../datasets/MIRFLICKR25K/filtered_topics/'
gradient_colors = ['#ff0000','#ff8000','#ffbf00','#ffff00','#00ff00']

num_topics = 400

num_pairs = 200000

# Read data
img_embeddings = {}
txt_embeddings = {}
file = open(data_file, "r")
print("Loading data ...")
for line in file:
    d = line.split(',')
    img_embeddings[d[0]] = np.fromstring(d[1], dtype=float, sep=' ')
    img_embeddings[d[0]] = img_embeddings[d[0]]  - min(img_embeddings[d[0]])
    img_embeddings[d[0]] = img_embeddings[d[0]] / sum(img_embeddings[d[0]])
    txt_embeddings[d[0]] = np.fromstring(d[2], dtype=float, sep=' ')
    txt_embeddings[d[0]] = txt_embeddings[d[0]] - min(txt_embeddings[d[0]])
    txt_embeddings[d[0]] = txt_embeddings[d[0]] / sum(txt_embeddings[d[0]])

print("Loading img topics...")
img_topics = {}
for file_name in glob.glob("/home/raulgomez/datasets/MIRFLICKR25K/filtered_topics/*.txt"):
    file = open(file_name, "r")
    lines = []
    for line in file:
        line = line.replace('\n','').replace('\t','').replace('\r','')
        lines.append(line)
    img_topics[file_name.split('/')[-1][:-4]] = lines[0].split(',') + lines[1].split(',')
    print img_topics[file_name.split('/')[-1][:-4]]
    file.close()


print("Computing pairs distances ...")
# Create pairs and compute distances between its img and text embeddings
distances = {}
x = 0
img_distances = np.zeros(num_pairs)
txt_distances = np.zeros(num_pairs)
colors = []

while x < num_pairs:
    id1 = random.choice(img_embeddings.keys())
    id2 = random.choice(img_embeddings.keys())
    img_dist = np.linalg.norm(img_embeddings[id1]-img_embeddings[id2])
    img_distances[x] = img_dist
    txt_dist = np.linalg.norm(txt_embeddings[id1]-txt_embeddings[id2])
    txt_distances[x] = txt_dist

    # Check if pair shares a hashtag
    img1_topics = img_topics[id1]
    img2_topics = img_topics[id2]
    matches = 0
    for t in img1_topics:
        if t in img2_topics:
            matches += 1
    if matches >= len(gradient_colors):
        matches = len(gradient_colors)-1
    colors.append(gradient_colors[matches])

    distances[str(id1) + "-" + str(id2)] = img_dist, txt_dist, matches
    x += 1


txt_distances = txt_distances / max(txt_distances)
img_distances = img_distances / max(img_distances)

# Plot: y axes img dist, x axes txt dist
print "Img embeddigns: Max distance: " + str(max(img_distances)) + " Min distance: " + str(min(img_distances))
print "Txt embeddigns: Max distance: " + str(max(txt_distances)) + " Min distance: " + str(min(txt_distances))

# colors_works = np.random.rand(num_pairs)
print len(colors)
print colors[0]
print len(txt_distances)
plt.scatter(txt_distances, img_distances, c=colors, alpha=0.5, s = 1)
# min = min(min(txt_distances),min(img_distances))
max = max(max(txt_distances),max(img_distances))
plt.plot((0,1), 'b--')
plt.xlim(0, max)
plt.ylim(0, max)
plt.title("Word2vec" )
plt.show()

