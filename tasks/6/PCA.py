from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


df = pd.read_csv('./datasets/weather_numeric.csv')

X = df.drop(columns=['dt_iso'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


pca = PCA()  # la den først velge alle komponenter
X_pca = pca.fit_transform(X_scaled)


explained_variance = pca.explained_variance_ratio_
print("Explained variance ratio per component:", explained_variance)

plt.plot(range(1, len(explained_variance)+1), explained_variance.cumsum(), marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Scree Plot")
plt.show()

pca = PCA(n_components=0.95)  # beholder komponenter som forklarer 95% av variansen
X_pca = pca.fit_transform(X_scaled)

print("Shape before PCA:", X_scaled.shape)
print("Shape after PCA:", X_pca.shape)

