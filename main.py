# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import silhouette_score
import tensorflow as tf
#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Input, Dense
import shap
from factor_analyzer import FactorAnalyzer
import warnings
warnings.filterwarnings('ignore')
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

# Set style for plots
plt.style.use('seaborn')
sns.set_palette("husl")

# %%
# Load and explore the data
df = pd.read_csv('africa_stock_5yrs.csv')
print("Data shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nData info:")
print(df.info())
print("\nBasic statistics:")
print(df.describe())

# Convert date to datetime
df['date'] = pd.to_datetime(df['date'])

# %%
# EDA and Visualization
plt.figure(figsize=(15, 10))

# Plot 1: Distribution of closing prices
plt.subplot(2, 2, 1)
sns.histplot(df['close'], kde=True)
plt.title('Distribution of Closing Prices')

# Plot 2: Volume distribution
plt.subplot(2, 2, 2)
sns.histplot(df['volume'], kde=True)
plt.title('Distribution of Trading Volume')

# Plot 3: Price trends for a sample stock
plt.subplot(2, 2, 3)
sample_stock = df[df['Name'] == 'MTN']
plt.plot(sample_stock['date'], sample_stock['close'])
plt.title('Price Trend for MTN')
plt.xlabel('Date')
plt.ylabel('Closing Price')

# Plot 4: Correlation heatmap
plt.subplot(2, 2, 4)
corr_matrix = df[['open', 'high', 'low', 'close', 'volume']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')

plt.tight_layout()
plt.show()

# %%
print("\nUnique stock names:")
print(df['Name'].unique())
print("Count:", len(df['Name'].unique()))

# %%
# Feature Engineering
def create_features(df):
    # Technical indicators
    df['daily_return'] = df.groupby('Name')['close'].pct_change()
    df['volatility'] = df.groupby('Name')['daily_return'].rolling(window=20).std().reset_index(0, drop=True)
    df['moving_avg_20'] = df.groupby('Name')['close'].rolling(window=20).mean().reset_index(0, drop=True)
    df['moving_avg_50'] = df.groupby('Name')['close'].rolling(window=50).mean().reset_index(0, drop=True)
    
    # Price momentum
    df['momentum'] = df.groupby('Name')['close'].pct_change(periods=5)
    
    # Volume features
    df['volume_ma'] = df.groupby('Name')['volume'].rolling(window=20).mean().reset_index(0, drop=True)
    df['volume_std'] = df.groupby('Name')['volume'].rolling(window=20).std().reset_index(0, drop=True)
    
    # Price range features
    df['price_range'] = (df['high'] - df['low']) / df['low']
    df['price_range_ma'] = df.groupby('Name')['price_range'].rolling(window=20).mean().reset_index(0, drop=True)
    
    return df

df = create_features(df)
df = df.dropna()

# %%
# Log Transformations of Various Features
df['log_close'] = np.log1p(df['close'])
df['log_volume'] = np.log1p(df['volume'])
df['log_open'] = np.log1p(df['open'])
df['log_high'] = np.log1p(df['high'])
df['log_low'] = np.log1p(df['low'])

# Plot the distributions before and after transformations
plt.figure(figsize=(15, 10))

# Closing Prices
plt.subplot(2, 3, 1)
sns.histplot(df['close'], kde=True)
plt.title('Original Closing Prices')

plt.subplot(2, 3, 4)
sns.histplot(df['log_close'], kde=True)
plt.title('Log-Transformed Closing Prices')

# Volume
plt.subplot(2, 3, 2)
sns.histplot(df['volume'], kde=True)
plt.title('Original Volume')

plt.subplot(2, 3, 5)
sns.histplot(df['log_volume'], kde=True)
plt.title('Log-Transformed Volume')

# Open Prices
plt.subplot(2, 3, 3)
sns.histplot(df['open'], kde=True)
plt.title('Original Open Prices')

plt.subplot(2, 3, 6)
sns.histplot(df['log_open'], kde=True)
plt.title('Log-Transformed Open Prices')

plt.tight_layout()
plt.show()

# %%
# Box Plot for Log-Transformed Closing Prices
plt.figure(figsize=(15, 8))
sns.boxplot(x='Name', y='log_close', data=df)
plt.xticks(rotation=90)
plt.title('Box Plot of Log-Transformed Closing Prices for Each Stock')
plt.show()
# %%
# Box Plot for Closing Prices of Each Stock
plt.figure(figsize=(15, 8))
sns.boxplot(x='Name', y='close', data=df)
plt.xticks(rotation=90)
plt.title('Box Plot of Closing Prices for Each Stock')
plt.show()

# %%
# Time Series Plot for a Sample Stock
sample_stocks = df['Name'].unique()[:] 
plt.figure(figsize=(15, 10))
for stock in sample_stocks:
    stock_data = df[df['Name'] == stock]
    plt.plot(stock_data['date'], stock_data['close'], label=stock)
plt.title('Time Series of Closing Prices for Sample Stocks')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.legend()
plt.show()

# %%
# Pair Plot for Feature Relationships
sns.pairplot(df[['daily_return', 'volatility', 'momentum', 'volume_ma', 'price_range']])
plt.suptitle('Pair Plot of Stock Features', y=1.02)
plt.show()

# %%
# Volatility Over Time for a Sample Stock
plt.figure(figsize=(15, 5))
sample_stock_data = df[df['Name'] == 'MTN']
plt.plot(sample_stock_data['date'], sample_stock_data['volatility'])
plt.title('Volatility Over Time for MTN')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.show()

# %%
# Volume vs. Closing Price
plt.figure(figsize=(10, 6))
sns.scatterplot(x='log_volume', y='log_close', data=df, alpha=0.5)
plt.title('Volume vs. Closing Price')
plt.xlabel('Volume')
plt.ylabel('Closing Price')
plt.show()

# %%
df.columns
# %%
# Feature Selection for Factor Analysis
features_for_fa = [
    'log_close',        # Price level
    'log_volume',       # Trading activity
    'daily_return',     # Price movement
    'volatility',       # Risk measure
    'momentum',         # Trend strength
    'price_range',      # Daily price range
    'volume_ma',        # Volume trend
    'moving_avg_20'     # Price trend
]

# Perform Factor Analysis
fa = FactorAnalyzer(n_factors=2, rotation='varimax')
fa.fit(df[features_for_fa].dropna())

# Get factor loadings
loadings = pd.DataFrame(fa.loadings_, columns=['Factor1', 'Factor2'], index=features_for_fa)
print("Factor Loadings:")
print(loadings)

# %%
# Plot Factor Loadings
plt.figure(figsize=(10, 6))
sns.heatmap(loadings, annot=True, cmap='coolwarm', center=0)
plt.title('Factor Loadings')
plt.show()

# %%
# Calculate Variance Explained
variance = fa.get_factor_variance()
print("\nVariance Explained:")
print(f"Factor 1: {variance[0][0]:.2%}")
print(f"Factor 2: {variance[0][1]:.2%}")
print(f"Total: {sum(variance[0]):.2%}")

# %%
# Plot Factor Scores
factor_scores = fa.transform(df[features_for_fa].dropna())
plt.figure(figsize=(10, 6))
plt.scatter(factor_scores[:, 0], factor_scores[:, 1], alpha=0.5)
plt.title('Factor Scores')
plt.xlabel('Factor 1')
plt.ylabel('Factor 2')
plt.show()


# %%
# Factor Analysis Interpretation
print("\nFactor Interpretation:")

# Get absolute loadings for better interpretation
abs_loadings = loadings.abs()

# Identify top features for each factor
factor1_top_features = abs_loadings['Factor1'].nlargest(3).index
factor2_top_features = abs_loadings['Factor2'].nlargest(3).index

print("\nFactor 1 (Market Risk/Price Movement) is most influenced by:")
for feature in factor1_top_features:
    loading = loadings.loc[feature, 'Factor1']
    print(f"- {feature}: {loading:.3f}")

print("\nFactor 2 (Trading Activity) is most influenced by:")
for feature in factor2_top_features:
    loading = loadings.loc[feature, 'Factor2']
    print(f"- {feature}: {loading:.3f}")

# %%
# Visualize Factor Contributions
plt.figure(figsize=(12, 6))

# Factor 1 Contributions
plt.subplot(1, 2, 1)
loadings['Factor1'].plot(kind='bar', color='skyblue')
plt.title('Factor 1: Market Risk/Price Movement')
plt.axhline(y=0, color='r', linestyle='-')
plt.xticks(rotation=45)
plt.tight_layout()

# Factor 2 Contributions
plt.subplot(1, 2, 2)
loadings['Factor2'].plot(kind='bar', color='lightgreen')
plt.title('Factor 2: Trading Activity')
plt.axhline(y=0, color='r', linestyle='-')
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()

# %%
# PCA Analysis
scaler = StandardScaler()
scaled_features = scaler.fit_transform(df[features_for_fa])
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_features)

# Plot PCA results
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
plt.title('PCA of Stock Features')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()

# %%
# Feature Selection for Anomaly Detection
features_to_keep = [
    'log_close',        # Transformed closing price
    'log_volume',       # Transformed volume
    'daily_return',     # Price movement
    'volatility',       # Price stability
    'momentum',         # Price trend
    'volume_ma',        # Volume trend
    'price_range',      # Daily price range
    'moving_avg_20'     # Short-term trend
]

# Select only the features we want to keep
X = df[features_to_keep].values

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nSelected features for anomaly detection:")
print(features_to_keep)

# %%

# %%
# Find Optimal Number of Clusters
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Plot Silhouette Scores
plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, marker='o')
plt.title('Silhouette Scores for Different Numbers of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.show()

# Find optimal number of clusters
optimal_k = k_range[np.argmax(silhouette_scores)]
print(f"\nOptimal number of clusters based on Silhouette Score: {optimal_k}")

# %%
# Update K-Means 
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Evaluate K-Means
silhouette_avg = silhouette_score(X_scaled, kmeans_labels)
print(f"Silhouette Score for K-Means: {silhouette_avg:.4f}")

# %%
# Detailed Cluster Analysis
print("\nDetailed Cluster Analysis:")

# Calculate cluster sizes
cluster_sizes = pd.Series(kmeans_labels).value_counts().sort_index()
print("\nCluster Sizes:")
for cluster, size in cluster_sizes.items():
    print(f"Cluster {cluster}: {size} stocks ({size/len(kmeans_labels):.1%})")

# Calculate mean values for each cluster
cluster_means = pd.DataFrame(X_scaled, columns=features_to_keep)
cluster_means['cluster'] = kmeans_labels
cluster_means = cluster_means.groupby('cluster').mean()

print("\nMean Feature Values by Cluster:")
print(cluster_means)

# %%
# Visualize Cluster Differences
plt.figure(figsize=(15, 10))

# Plot 1: Feature Importance by Cluster
plt.subplot(2, 2, 1)
cluster_means.T.plot(kind='bar')
plt.title('Feature Values by Cluster')
plt.xticks(rotation=45)
plt.legend([f'Cluster {i}' for i in range(optimal_k)])
plt.tight_layout()

# Plot 2: PCA Visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

plt.subplot(2, 2, 2)
scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                     c=kmeans_labels, cmap='viridis', alpha=0.5)
plt.colorbar(scatter, label='Cluster')
plt.title('K-Means Clusters (PCA Visualization)')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')

# Plot 3: Volatility Distribution by Cluster
plt.subplot(2, 2, 3)
# Create a DataFrame with the original data and cluster labels
plot_data = pd.DataFrame(X_scaled, columns=features_to_keep)
plot_data['cluster'] = kmeans_labels
sns.boxplot(x='cluster', y='volatility', data=plot_data)
plt.title('Volatility Distribution by Cluster')

# Plot 4: Volume Distribution by Cluster
plt.subplot(2, 2, 4)
sns.boxplot(x='cluster', y='log_volume', data=plot_data)
plt.title('Volume Distribution by Cluster')

plt.tight_layout()
plt.show()

# %%
# Analyze Stock Characteristics by Cluster
cluster_analysis = pd.DataFrame(X_scaled, columns=features_to_keep)
cluster_analysis['cluster'] = kmeans_labels
cluster_analysis['Name'] = df['Name'].values

# Print sample stocks from each cluster
print("\nSample Stocks from Each Cluster:")
for cluster in range(optimal_k):
    print(f"\nCluster {cluster} Stocks:")
    cluster_stocks = cluster_analysis[cluster_analysis['cluster'] == cluster]['Name'].unique()
    print(f"Number of unique stocks: {len(cluster_stocks)}")
    print("Sample stocks:", cluster_stocks[:5])  # Show first 5 stocks

# %%
# Calculate Cluster Statistics
print("\nCluster Statistics:")
for cluster in range(optimal_k):
    cluster_data = cluster_analysis[cluster_analysis['cluster'] == cluster]
    print(f"\nCluster {cluster}:")
    print(f"Average Volatility: {cluster_data['volatility'].mean():.4f}")
    print(f"Average Daily Return: {cluster_data['daily_return'].mean():.4f}")
    print(f"Average Volume: {np.exp(cluster_data['log_volume']).mean():.2f}")
    print(f"Average Price Range: {cluster_data['price_range'].mean():.4f}")

# %%
# Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_labels = iso_forest.fit_predict(X_scaled)

# %%
# One-Class SVM
oc_svm = OneClassSVM(nu=0.1)
svm_labels = oc_svm.fit_predict(X_scaled)

# %%
# Autoencoder
input_dim = X_scaled.shape[1]
encoding_dim = 2

input_layer = Input(shape=(input_dim,))
encoder = Dense(encoding_dim, activation="relu")(input_layer)
decoder = Dense(input_dim, activation="sigmoid")(encoder)

autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.compile(optimizer='adam', loss='mse')

# Train autoencoder
autoencoder.fit(X_scaled, X_scaled,
                epochs=50,
                batch_size=32,
                shuffle=True,
                validation_split=0.2,
                verbose=0)

# Get reconstruction error
reconstructions = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)

# %%
# Gaussian Mixture Model (GMM)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)
gmm_scores = -gmm.score_samples(X_scaled)  # Negative log-likelihood as anomaly score

# %%
# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

# %%
# Model Evaluation and Visualization
plt.figure(figsize=(20, 15))

# Plot 1: K-Means clusters
plt.subplot(3, 2, 1)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-Means Clustering')

# Plot 2: Isolation Forest anomalies
plt.subplot(3, 2, 2)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=iso_labels, cmap='coolwarm')
plt.title('Isolation Forest Anomalies')

# Plot 3: One-Class SVM anomalies
plt.subplot(3, 2, 3)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=svm_labels, cmap='coolwarm')
plt.title('One-Class SVM Anomalies')

# Plot 4: Autoencoder reconstruction error
#plt.subplot(3, 2, 4)
#plt.scatter(pca_result[:, 0], pca_result[:, 1], c=mse, cmap='viridis')
#plt.colorbar(label='Reconstruction Error')
#plt.title('Autoencoder Reconstruction Error')

# Plot 5: GMM Anomalies
plt.subplot(3, 2, 5)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=gmm_scores, cmap='viridis')
plt.colorbar(label='GMM Anomaly Score')
plt.title('GMM Anomaly Detection')

# Plot 6: DBSCAN Clusters
plt.subplot(3, 2, 6)
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=dbscan_labels, cmap='viridis')
plt.title('DBSCAN Clustering')

plt.tight_layout()
plt.show()

# %%
# Print number of anomalies detected by each method
print("\nNumber of anomalies detected by each method:")
print(f"K-Means: {len(np.where(kmeans_labels == -1)[0])}")
print(f"Isolation Forest: {len(np.where(iso_labels == -1)[0])}")
print(f"One-Class SVM: {len(np.where(svm_labels == -1)[0])}")
print(f"GMM (top 5%): {len(np.where(gmm_scores > np.percentile(gmm_scores, 95))[0])}")
print(f"DBSCAN (noise points): {len(np.where(dbscan_labels == -1)[0])}")

# %%
# SHAP Analysis for interpretability
explainer = shap.KernelExplainer(oc_svm.decision_function, X_scaled[:100])
shap_values = explainer.shap_values(X_scaled[:100])

# Plot SHAP values
shap.summary_plot(shap_values, X_scaled[:100], feature_names=features_to_keep)
