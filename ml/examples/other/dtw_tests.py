import numpy as np
import tests.dtw as dtw
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClusterMixin

def title():
    return 'DTW tests'

def main():

    #test1()
    test2()

    return


def test1():
    x = np.array([0, 0, 1, 1, 2, 4, 2, 1, 2, 0]).reshape(-1, 1)
    y = np.array([1, 1, 1, 2, 2, 2, 2, 3, 2, 0]).reshape(-1, 1)

    plt.plot(x)
    plt.plot(y)

    dist, cost, acc, path = dtw.dtw(x, y, dist=lambda x, y: np.linalg.norm(x - y, ord=2))

    print(dist)

    plt.imshow(acc.T, origin='lower', cmap=plt.cm.gray, interpolation='nearest')
    plt.plot(path[0], path[1], 'w')
    plt.xlim((-0.5, acc.shape[0]-0.5))
    plt.ylim((-0.5, acc.shape[1]-0.5))

    plt.show()


def test2():

    np.random.seed(9)

    n1 = np.random.normal(size=[50, 2])
    n1[0, 0] = 0.0
    n1[0, 1] = 0.0
    n2 = np.random.normal(2.5, size=[50, 2])
    n2[0, 1] = 3.0
    n2[0, 1] = 3.0

    X = np.concatenate([n1, n2])
    print(X.shape)

    y = np.array([None]*100)
    y[0] = 0
    y[1] = 0
    y[2] = 0
    y[50] = 1
    y[51] = 1
    y[52] = 1

    clust = SimpleClusterizer(lambda x, y: np.linalg.norm(x - y), random_state=9)
    clust.fit(X, y)

    from matplotlib.colors import ListedColormap

    colors = ListedColormap(['red', 'blue'])
    light_colors = ListedColormap(['lightcoral', 'lightblue'])

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

    predictions = clust.predict(np.c_[xx.ravel(), yy.ravel()])
    mesh_predictions = np.array(predictions).reshape(xx.shape)
    plt.pcolormesh(xx, yy, mesh_predictions, cmap = light_colors)

    plt.scatter(n1[:, 0], n1[:, 1])
    plt.scatter(n2[:, 0], n2[:, 1])

    #log = np.array(clust.log[:2])
    #plt.scatter(log[:, 0], log[:, 1], s=70, color='black')

    plt.show()

    return

class SimpleClusterizer(BaseEstimator, ClusterMixin):
    def __init__(self, dist, random_state=None):
        self.dist = dist
        self.random_state = random_state
        self._labels = None
        self._clusters = None


    def fit(self, X, y):
        labels = np.unique(y[y != np.array(None)])

        clusters = { l: list(X[y==l]) for l in labels }

        X_unm = X[y==None]
        np.random.seed(self.random_state)
        np.random.shuffle(X_unm)

        self.log = []

        for x in X_unm:
            label = None
            dist = np.inf
            for l in labels:
                cluster = clusters[l]
                d = self._get_dist(x, cluster)
                if d < dist:
                    dist = d
                    label = l
            clusters[label].append(x)
            self.log.append(x)

        self._clusters = clusters
        self._labels = labels


    def predict(self, X):
        y = []
        for x in X:
            label = None
            dist = np.inf
            for l in self._labels:
                cluster = self._clusters[l]
                d = self._get_dist(x, cluster)
                if d < dist:
                    dist = d
                    label = l
            y.append(label)
        return y


    def _get_dist(self, x, cluster):
        dist = np.inf
        for xc in cluster:
            d = self.dist(x, xc)
            if d < dist:
                dist = d
        #dist = np.array([self.dist(x, xc) for xc in cluster]).mean()
        return dist


