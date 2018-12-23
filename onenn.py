import math
import numpy as np
import matplotlib.pyplot as plt

# formulate two distribution with random noise
np.random.seed(0)
sample_num = np.arange(20)
x1 = 2*sample_num + 3*np.random.randn(20)
x2 = 3*sample_num + 40 + 2*np.random.randn(20)

# extract one instance from each distribution with label
labeled_data = [[[15, x1[15]],0], [[5, x2[5]],1]]

# make array of other data as unlabeled
unlabeled_data = [[i, x1[i]] for i in sample_num if i != 15]
unlabeled_data.extend([[i, x2[i]] for i in sample_num if i != 5])

unlabeled_data = np.array(unlabeled_data)

# shuffle unlabeled
np.random.shuffle(unlabeled_data)

# plot current data distribution
plt.scatter(unlabeled_data[:, 0], unlabeled_data[: , 1], marker='x', color='g')
plt.plot(15, x1[15], marker='o', color='r')
plt.plot(5, x2[5], marker='o', color='b')
plt.show()

# for each single unlabeled data
for single_u in unlabeled_data:

    # calculate distance to nearby labeled data
    distances_a = []
    distances_b = []

    # for all labeeld data
    for single_l in labeled_data:

        # calcualte L2 distance
        dist = math.sqrt((single_u[0]-single_l[0][0])**2 + (single_u[1]-single_l[0][1])**2)

        # calcualte distances with labels
        if single_l[1] == 0:
            distances_a.append(dist)
        else:
            distances_b.append(dist)

    # compare distances and give label of minimum fistnace
    if min(distances_a) < min(distances_b):
        labeled_data.append([single_u, 0])

    elif min(distances_a) > min(distances_b):
        labeled_data.append([single_u, 1])

    # break tie
    else:
        labeled_data.append([single_u, int(round(np.random.rand(1)[0]))])

# plot the result
pseudo_x1 = np.array([i[0] for i in labeled_data if i[1] == 0])
pseudo_x2 = np.array([i[0] for i in labeled_data if i[1] == 1])

plt.scatter(pseudo_x1[:, 0], pseudo_x1[:,1], marker="x", color="r")
plt.scatter(pseudo_x2[:, 0], pseudo_x2[:,1], marker="o", color="b")
plt.show()