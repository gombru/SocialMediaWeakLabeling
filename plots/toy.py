import matplotlib.pyplot as plt

x = [3,3,4,6,3,6,6,20]
y = range(len(x))
plt.figure(figsize=(10,10))
plt.scatter(x, y, alpha=0.5)
plt.title("Word2vec" )
plt.xlabel("txt distance")
plt.ylabel("img distance")

plt.show()
