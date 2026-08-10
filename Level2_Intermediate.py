"""
Codveda Technologies - Data Analysis Internship
LEVEL 2 (INTERMEDIATE)
Task 1: Regression Analysis
Task 3: Clustering Analysis (K-Means)

Author: Sandra Uzoamaka Ani
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

sns.set_style("whitegrid")

# ---------------------------------------------------------
# TASK 1: REGRESSION ANALYSIS
# Predict median home value (MEDV) from the other features
# ---------------------------------------------------------
print("=" * 60)
print("TASK 1: REGRESSION ANALYSIS")
print("=" * 60)

df = pd.read_csv('/mnt/user-data/outputs/house_data_cleaned.csv')

X = df.drop(columns=['MEDV'])
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nR-squared: {r2:.3f}  (model explains {r2*100:.1f}% of the variance in home prices)")
print(f"RMSE: {rmse:.2f}  (typical prediction is off by ~${rmse*1000:,.0f})")

coefs = pd.Series(model.coef_, index=X.columns).sort_values(key=abs, ascending=False)
print(f"\nTop feature coefficients (impact on price per unit increase):\n{coefs.head(6).round(2)}")

plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='steelblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Price ($1000s)')
plt.ylabel('Predicted Price ($1000s)')
plt.title(f'Regression: Actual vs Predicted (R² = {r2:.3f})')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/plots/level2_regression.png', dpi=150)
plt.close()

# ---------------------------------------------------------
# TASK 3: CLUSTERING ANALYSIS (K-MEANS)
# Group similar flowers by feature similarity
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 3: CLUSTERING ANALYSIS (K-MEANS)")
print("=" * 60)

iris = pd.read_csv('/mnt/user-data/uploads/1__iris.csv')
print(f"\nColumns: {list(iris.columns)}")

feature_cols = iris.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in feature_cols if c.lower() != 'id']
X_iris = iris[feature_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_iris)

# Elbow method
inertias = []
k_range = range(1, 9)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(k_range, inertias, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/plots/level2_elbow.png', dpi=150)
plt.close()

# Fit final model with k=3 (elbow point, and matches the 3 known iris species)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
iris['cluster'] = kmeans.fit_predict(X_scaled)

print(f"\nCluster sizes:\n{iris['cluster'].value_counts().sort_index()}")

if 'species' in [c.lower() for c in iris.columns]:
    species_col = [c for c in iris.columns if c.lower() == 'species'][0]
    crosstab = pd.crosstab(iris['cluster'], iris[species_col])
    print(f"\nCluster vs actual species (sanity check):\n{crosstab}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_iris.iloc[:, 0], y=X_iris.iloc[:, 1],
                 hue=iris['cluster'], palette='Set2', s=70)
plt.xlabel(feature_cols[0])
plt.ylabel(feature_cols[1])
plt.title('K-Means Clusters (k=3)')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/plots/level2_clusters.png', dpi=150)
plt.close()

print("\nPlots saved -> plots/level2_regression.png, plots/level2_elbow.png, plots/level2_clusters.png")
