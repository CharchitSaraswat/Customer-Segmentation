import numpy as np
import pandas as pd
import visuals as vs
from IPython.display import display # Allows the use of display() for DataFrames
from sklearn.cross_validation import train_test_split as Tts
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.mixture import GMM
from sklearn.metrics import silhouette_score

try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))
except:
    print("Dataset could not be loaded. Is the dataset missing?")

# Feature Relevance:
feature_dropped='Fresh'
new_data = data.drop(feature_dropped,axis=1)
labels=data[feature_dropped]
X_train, X_test, y_train, y_test = Tts(new_data, labels, test_size=0.25, random_state=30)
regressor = DecisionTreeRegressor(random_state=30)
regressor.fit(X_train,y_train)
score = regressor.score(X_test,y_test)
print(score)

#Visualize Feature Distribution
pd.plotting.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

#Logarithmic Feature Scaling
log_data = np.log(data)
log_samples = np.log(samples)
pd.plotting.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde');

#Outlier Detection
for feature in log_data.keys():
    Q1 = np.percentile(log_data[feature],25)
    Q3 = np.percentile(log_data[feature],75)
    step = (Q3-Q1)*1.5
    print("Data points considered outliers for the feature '{}':".format(feature))
    display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])
outliers  = [65,66,75,128,154]
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)

#Feature Transformation: PCA
pca = PCA(n_components=len(good_data.columns)).fit(good_data)
pca_samples = pca.transform(log_samples)
explained_var=pca.explained_variance_ratio_
totl=0
explained_var2=sum([explained_var[i] for i in range(2)])
explained_var4=sum([explained_var[i] for i in range(4)])
print('Total Variance from first 2 components:',explained_var2)
print('Total Variance from first 2 components:',explained_var4)
pca_results = vs.pca_results(good_data, pca)

#Dimensionality Reduction
pca = PCA(n_components=2).fit(good_data)
reduced_data = pca.transform(good_data)
pca_samples = pca.transform(log_samples)
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])

#Biplot
vs.biplot(good_data, reduced_data, pca)

#K-Means Clustering
#Test for various number of clusters and check scores.
clusterer = GMM(n_components=2).fit(reduced_data)
preds = clusterer.predict(reduced_data)
centers = clusterer.means_
sample_preds = clusterer.predict(pca_samples)
score = silhouette_score(reduced_data,preds)
print(score)

#Cluster Plot
vs.cluster_results(reduced_data, preds, centers, pca_samples)

#Data Recovery
log_centers = pca.inverse_transform(centers)
true_centers = np.exp(log_centers)
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)

#Sample point predictions:
for i, pred in enumerate(sample_preds):
    print("Sample point", i, "predicted to be in Cluster", pred)

#Clustering based on Channel data
vs.channel_results(reduced_data, outliers, pca_samples)
