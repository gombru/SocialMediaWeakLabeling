file = "/home/raulgomez/datasets/SocialMedia/val_InstaCities1M_divbymax.txt"
outfile = open("/home/raulgomez/datasets/SocialMedia/val_indices.txt" ,'w')

for line in open(file):
    line = line.split(',')[0] + '.jpg'
    outfile.write(line + '\n')

outfile.close()