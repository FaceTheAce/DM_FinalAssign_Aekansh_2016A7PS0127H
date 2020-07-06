#Aekansh 2016A7PS0127H
from itertools import cycle
from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.cluster import Birch
#from sklearn.datasets import make_blobs
from scipy.io import arff
import pandas as pd
#from sklearn.cluster import Birch
from sklearn.utils import check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.metrics.pairwise import pairwise_distances_argmin
from memory_profiler import profile


import os

n_decompositon = 1000  # divide the array 'reduced_distance' into 1000 parts along the axis=0

class BirchChunked(Birch):
    @profile
    def predict(self, X):
        # the original code
        X = check_array(X, accept_sparse='csr')
        self._check_fit(X)
        return self.subcluster_labels_[pairwise_distances_argmin(X, self.subcluster_centers_)]


class NewBirch(Birch):
    @profile
    def predict(self, X):
        X = check_array(X, accept_sparse='csr')
        self._check_fit(X)
        # assume that the matrix is dense
        argmin_list = np.array([], dtype=np.int)
        interval = int(np.ceil(X.shape[0] / n_decompositon))
        for index in range(0, n_decompositon - 1):
            lb = index * interval
            ub = (index + 1) * interval
            reduced_distance = safe_sparse_dot(X[lb:ub, :], self.subcluster_centers_.T)
            reduced_distance *= -2
            reduced_distance += self._subcluster_norms
            argmin_list = np.append(argmin_list, np.argmin(reduced_distance, axis=1))

        lb = (n_decompositon - 1) * interval
        reduced_distance = safe_sparse_dot(X[lb:X.shape[0], :], self.subcluster_centers_.T)
        reduced_distance *= -2
        reduced_distance += self._subcluster_norms
        argmin_list = np.append(argmin_list, np.argmin(reduced_distance, axis=1))

        return self.subcluster_labels_[argmin_list]


# =============================================================================
# 
# =============================================================================

#get path of the python file
path = os.getcwd()
#get parent directory of the current directory
root = os.path.abspath(os.path.join(path, os.pardir))

col = ['x','y']
df = pd.DataFrame(columns=col)

files = [f for f in os.listdir( root + r'\Dataset') ]


#read all the individual files and get them into a single dataframe
for filename in files:  
    dest = root + r'\Dataset'
    dest= dest + '//'+  filename
    data = arff.loadarff(dest)
    df_sub = pd.DataFrame(data[0])

#change the names of variables into a uniform (x,y) coordinate

    df_sub = df_sub.rename(columns={'a0': 'x', 'a1' : 'y'})
    df_sub = df_sub[['x','y']]
    df = df.append(df_sub, ignore_index = True)

#drop duplicate rows
df.drop_duplicates(keep=False,inplace=True)

#convert Dataframe into numpy array
X = df.to_numpy()

# Use all colors that matplotlib provides by default.
colors_ = cycle(colors.cnames.keys())

fig = plt.figure(figsize=(8, 4))
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.1, top=0.9)

# Compute clustering with Birch with and without the final clustering step
# and plot.
birch_models = [BirchChunked(threshold=1.7, n_clusters=None),
            
#                NewBirch(threshold=1.7, n_clusters=None)]
                ]
final_step = ['with chunked predict(X)'
#              ,'with new predict(X) method'           
                ]
for ind, (birch_model, info) in enumerate(zip(birch_models, final_step)):
    t = time()
    birch_model.fit(X)
    time_ = time() - t
    print("Birch %s as the final step took %0.2f seconds" % (
          info, (time() - t)))

    # Plot result
    labels = birch_model.labels_
    centroids = birch_model.subcluster_centers_
    n_clusters = np.unique(labels).size
    print("n_clusters : %d" % n_clusters)

    ax = fig.add_subplot(1, 3, ind + 1)
    for this_centroid, k, col in zip(centroids, range(n_clusters), colors_):
        mask = labels == k
        ax.scatter(X[mask, 0], X[mask, 1],
                   c='w', edgecolor=col, marker='.', alpha=0.5)

        ax.scatter(this_centroid[0], this_centroid[1], marker='+',
                   c='k', s=25)
    #show full plot
#    ax.set_autoscaley_on(True)
#    ax.set_title('Birch %s' % info)
#    plt.show()       
    #show zoomed in version of the plot
    ax.set_ylim([-25, 25])
    ax.set_xlim([-25, 25])
    ax.set_autoscaley_on(False)
    ax.set_title('Birch %s' % info)
    plt.show()
