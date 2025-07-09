import numpy as np
import matplotlib.pyplot as plt

def inertia(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    inertia_value = np.sum(np.min(distances, axis=1) ** 2)
    return inertia_value

def mapCentroids(centroids):
    r = ''
    for i, centroid in enumerate(centroids):
        r += f'C{i}: {" ".join(f"{x:8.3f}" for x in centroid)}\n'
    return r

def distancesDf(distances, labels=None):
    import pandas as pd
    df = pd.DataFrame(distances, columns=[f'C{i}' for i in range(distances.shape[1])])
    if labels: df['Label'] = labels
    return df.round(3)
    

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-8, labels=None, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None
        self.labels = labels
        self.inertia_ = None
        self.random_state = random_state

    def fit(self, X, centroids=None, verbose=False):
        n_samples, _ = X.shape
        # Randomly initialize centroids
        if self.random_state is not None:
            np.random.seed(self.random_state)
        if centroids is None:
            random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            self.centroids = X[random_indices]
        else:
            self.centroids = centroids

        for i in range(self.max_iter):
            # Assign clusters based on closest centroid
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            # Calculate new centroids
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            # Check for convergence
            if np.linalg.norm(new_centroids - self.centroids) < self.tol:
                break
            self.centroids = new_centroids
            if verbose: 
                print(f"Iteration {i + 1}")
                print(f"Distances:\n{distancesDf(distances, self.labels)}")
                print(f"Centroids:\n{mapCentroids(self.centroids)}\n")
        self.inertia_ = inertia(X, self.centroids)
        if verbose: print(f"Converged after {i + 1} iterations.")

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)