import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# App Title
st.title("🧠 Customer Segmentation using K-Means Clustering")

# Load Dataset
st.subheader("📂 Loading Mall Customers Dataset")
df = pd.read_csv(r"F:\AWFERA\Machine learning\UnsupervisedLearningAlgorithmsProject\Mall_Customers.csv")
st.write(df.head())

# Show Column Names
st.write("📌 Column Names:", df.columns.tolist())

# Null Value Check
st.subheader("🔍 Null Value Check")
st.write(df.isnull().sum())

# Encode Gender Column (if exists)
if 'Gender' in df.columns:
    st.subheader("🧬 Encoding Gender Column (Male=0, Female=1)")
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    st.write(df.head())

# Feature Selection
features = ['Annual Income (k$)', 'Spending Score (1-100)']
X = df[features]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow Method
st.subheader("📉 Elbow Method for Optimal Clusters")
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

fig_elbow = plt.figure()
plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("WCSS (Inertia)")
st.pyplot(fig_elbow)

# Select Number of Clusters
st.subheader("🔧 Select Number of Clusters")
k = st.slider("Select k (number of clusters):", min_value=2, max_value=10, value=5)

# Apply KMeans
kmeans = KMeans(n_clusters=k, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Add cluster results to DataFrame
df['Cluster'] = y_kmeans

# Cluster Summary
st.subheader("📊 Cluster Summary (Averages)")
st.write(df.groupby('Cluster')[features].mean())

# Visualize Clusters
st.subheader("🧩 Cluster Visualization")
fig_cluster = plt.figure()
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, cmap='rainbow', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='black', marker='X', label='Centroids')
plt.xlabel("Annual Income (scaled)")
plt.ylabel("Spending Score (scaled)")
plt.legend()
plt.title(f"{k} Customer Segments")
st.pyplot(fig_cluster)

# Silhouette Score
score = silhouette_score(X_scaled, y_kmeans)
st.success(f"✅ Silhouette Score for k={k}: {score:.3f}")

# Clustered Dataset
st.subheader("📋 Clustered Data Preview")
st.write(df.head())

# Predict Cluster for New Customer
st.subheader("🧪 Predict Cluster for New Customer")

income = st.number_input("Enter Annual Income (k$):", min_value=0)
score = st.number_input("Enter Spending Score (1–100):", min_value=0, max_value=100)

if st.button("Predict Cluster"):
    new_data = [[income, score]]
    new_data_scaled = scaler.transform(new_data)
    cluster = kmeans.predict(new_data_scaled)[0]
    st.success(f"The new customer belongs to Cluster: {cluster}")

    # Plot New Customer on Graph
    fig_new = plt.figure()
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, cmap='rainbow', alpha=0.6)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='black', marker='X', label='Centroids')
    plt.scatter(new_data_scaled[0][0], new_data_scaled[0][1], c='red', marker='*', s=300, label='New Customer')
    plt.xlabel("Annual Income (scaled)")
    plt.ylabel("Spending Score (scaled)")
    plt.legend()
    plt.title("New Customer on Cluster Graph")
    st.pyplot(fig_new)
