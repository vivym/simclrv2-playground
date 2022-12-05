import pandas as pd
from sklearn import metrics
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
import numpy as np

labels_file='./labels.csv'
labels_wen=pd.read_csv(labels_file,header=None,index_col=None)
print("labels_wen", labels_wen)
labels=np.array(labels_wen)
print("labels", labels.shape)
labels=labels.ravel()
print("labels", labels.shape)
print("labels", labels)
exit(0)
feature_file='H:\论文发表\李培峦老师\栗莹学姐\SimCLR\DLPFC_151673\stMVC/data.csv'
feature=pd.read_csv(feature_file,header=None,index_col=None)
feature=np.array(feature)
#kmeans=KMeans(n_clusters=8)
kmeans=KMeans(n_clusters=8,init='k-means++')
preds_k=kmeans.fit_predict(feature)
np.savetxt ('pred.csv', preds_k, delimiter=',')
kk=metrics.adjusted_rand_score(labels,preds_k)
kk1=normalized_mutual_info_score(labels,preds_k)
print('kmeans','ARI:',kk,'NMI:',kk1)