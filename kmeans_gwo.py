# Creator : Said Al Afghani Edsa
#
# Date : 21/05/2023
#
#

import numpy as np
from sklearn.datasets import make_blobs

class KMeans:
    def __init__(self, n_clusters, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def fit(self, X, initial_centroids=None):
        if initial_centroids is None:
            self.centroids = X[np.random.choice(range(X.shape[0]), self.n_clusters, replace=False)]
        else:
            self.centroids = initial_centroids

        for _ in range(self.max_iter):
            labels = np.array([np.argmin(np.linalg.norm(x - self.centroids, axis=1)) for x in X])
            new_centroids = np.array([X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 else self.centroids[i] for i in range(self.n_clusters)])

            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break
            self.centroids = new_centroids

        return self

    def predict(self, X):
        return [np.argmin(np.linalg.norm(x - self.centroids, axis=1)) for x in X]

class GWO:
    def __init__(self, n_clusters, n_wolves, X, max_iter=100, inertia_weight=0.5, tol=1e-4):
        self.n_clusters = n_clusters
        self.n_wolves = n_wolves
        self.X = X
        self.wolves = [KMeans(n_clusters).fit(X) for _ in range(n_wolves)]
        self.alpha, self.beta, self.delta = sorted(self.wolves, key=self._fitness)[:3]
        self.max_iter = max_iter
        self.inertia_weight = inertia_weight
        self.tol = tol

    def _fitness(self, wolf):
        labels = np.array(wolf.predict(self.X))
        return sum([np.linalg.norm(self.X[labels == i] - wolf.centroids[i]) for i in range(self.n_clusters)])

    def _update_position(self, wolf, iteration):
        a = 2 - iteration * ((2) / self.max_iter)
        r1, r2 = np.random.rand(), np.random.rand()
        A = 2 * a * r1 - a
        C = 2 * r2
        D_alpha = abs(C * self.alpha.centroids - wolf.centroids)
        X_alpha = self.alpha.centroids - A * D_alpha
        r1, r2 = np.random.rand(), np.random.rand()
        A = 2 * a * r1 - a
        C = 2 * r2
        D_beta = abs(C * self.beta.centroids - wolf.centroids)
        X_beta = self.beta.centroids - A * D_beta
        r1, r2 = np.random.rand(), np.random.rand()
        A = 2 * a * r1 - a
        C = 2 * r2
        D_delta = abs(C * self.delta.centroids - wolf.centroids)
        X_delta = self.delta.centroids - A * D_delta
        new_position = (X_alpha*0.5 + X_beta*0.3 + X_delta*0.2) / 3 #can be modified
        wolf.fit(self.X, initial_centroids=self.inertia_weight * wolf.centroids + (1 - self.inertia_weight) * new_position)
        return wolf

    def optimize(self):
        for iteration in range(self.max_iter):
            for i, wolf in enumerate(self.wolves):
                self.wolves[i] = self._update_position(wolf, iteration)
            self.alpha, self.beta, self.delta = sorted(self.wolves, key=self._fitness)[:3]
            if np.max([np.linalg.norm(wolf.centroids - self.alpha.centroids) for wolf in self.wolves]) < self.tol:
                break
        return self.alpha

# Membuat data acak
X, _ = make_blobs(n_samples=300, centers=4, n_features=2)

# Jumlah cluster yang diinginkan
n_clusters = 4

# Jumlah wolves dalam GWO
n_wolves = 10

# Maksimum iterasi untuk GWO
max_iter = 100

# Inertia weight untuk GWO
inertia_weight = 0.5

# Membuat instance dari GWO
gwo = GWO(n_clusters, n_wolves, X, max_iter, inertia_weight)

# Menjalankan optimasi
best_wolf = gwo.optimize()

# Mendapatkan centroids terbaik
best_centroids = best_wolf.centroids

# Membuat instance dari KMeans menggunakan centroids terbaik
kmeans = KMeans(n_clusters)
kmeans.centroids = best_centroids

# Prediksi label data
labels = kmeans.predict(X)

# Cetak label
print(labels)





