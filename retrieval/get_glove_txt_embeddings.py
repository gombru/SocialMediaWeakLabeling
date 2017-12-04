# Retrieves nearest images given a text query and saves them in an given folder

import text2topics
import glove

out_file = 'top5_SM.txt'
out_file_path = '../../../datasets/SocialMedia/models/glove/txt_embeddings/' + out_file
model_name = 'glove_model_InstaCities1M.model'
model_path = '../../../datasets/SocialMedia/models/glove/' + model_name
model = glove.Glove.load(model_path)
num_topics = 400

# Do default queryes
q = []
w = [] # Weights per word (can be negative)

# # Simple
q.append('car')
q.append('skyline')
q.append('bike')

q.append('sunrise')
q.append('snow')
q.append('rain')

q.append('icecream')
q.append('cake')
q.append('pizza')

q.append('woman')
q.append('man')
q.append('kid')

# Complex
q.append('yellow car')
q.append('skyline night')
q.append('bike park')

q.append('sunrise beach')
q.append('snow ski')
q.append('rain umbrella')

q.append('icecream beach')
q.append('chocolate cake')
q.append('pizza wine')

q.append('woman bag')
q.append('man boat')
q.append('kid dog')

file = open(out_file_path,'w')

for e,cur_q in enumerate(q):
    if len(cur_q.split(' ')) == 1:
        topics = text2topics.glove(cur_q, '1', model, num_topics)
    if len(cur_q.split(' ')) == 2:
        topics = text2topics.glove(cur_q, '0.5 0.5', model, num_topics)
    file.write(cur_q)
    for t in topics:
        file.write(',' + str(t))











