import numpy as np

# Embeddings
input = "/home/raulgomez/projects/TSNE/aux_data/SocialMedia_glove_embeddings.txt"
out = "/home/raulgomez/projects/TSNE/aux_data/SocialMedia_glove_embeddings_divbymax.txt"

with open(out, 'w') as outfile:
    with open(input) as infile:
        for line in infile:
            data = line.split(',')
            id = data[0]
            values = np.asarray(data[1:], dtype=np.float32)
            values = values - min(values)
            values = values / max(values)
            cur_out = id
            for v in values:
                cur_out = cur_out + ',' + str(v)
            outfile.write(cur_out + '\n')
outfile.close()
