import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans


def main():
    df = pd.read_csv("labels.csv")

    ids = np.asarray(df["ID"])
    labels = np.asarray(df["Cluster"])
    print(ids.shape)
    print(labels.shape, labels.dtype)

    features = []
    for idx, label in zip(ids, labels):
        feature = np.load(f"features/{idx}40.jpeg.npy")
        features.append(feature)

    features = np.stack(features, 0)
    print(features.shape)

    kmeans = KMeans(n_clusters=20, init="k-means++")
    preds_k = kmeans.fit_predict(features)
    print("preds_k", preds_k.shape, preds_k)

    kk = metrics.adjusted_rand_score(labels, preds_k)
    kk1 = normalized_mutual_info_score(labels, preds_k)
    print("kmeans", "ARI:", kk, "NMI:",kk1)


if __name__ == "__main__":
    main()
