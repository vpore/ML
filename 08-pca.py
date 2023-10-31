import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer().data
print(data.shape)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
print(scaled_data)

n_components = 2
pca = PCA(n_components=n_components)

pca.fit(scaled_data)

transformed_data = pca.transform(scaled_data)

explained_variance_ratio = pca.explained_variance_ratio_
print("Explained Variance Ratio:", explained_variance_ratio)

principal_components = pca.components_
print("Principal Components:", principal_components)


plt.figure(figsize=(10, 8))

plt.scatter(transformed_data[:, 0], transformed_data[:, 1],
            c=load_breast_cancer().target,
            cmap='plasma')

plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()