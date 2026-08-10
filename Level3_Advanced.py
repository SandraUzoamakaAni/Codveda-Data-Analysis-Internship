"""
Codveda Technologies - Data Analysis Internship
LEVEL 3 (ADVANCED)
Task 1: Predictive Modeling (Classification) - Customer Churn
Task 2: Building Dashboards with Power BI/Tableau (data prep)

Author: Sandra Uzoamaka Ani
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sns.set_style("whitegrid")

# ---------------------------------------------------------
# TASK 1: PREDICTIVE MODELING (CLASSIFICATION)
# ---------------------------------------------------------
print("=" * 60)
print("TASK 1: CHURN CLASSIFICATION")
print("=" * 60)

# Codveda gave this dataset pre-split 80/20 - using it as intended
train = pd.read_csv('/mnt/user-data/uploads/churn-bigml-80.csv')
test = pd.read_csv('/mnt/user-data/uploads/churn-bigml-20.csv')

print(f"\nTrain: {train.shape}, Test: {test.shape}")
print(f"Churn rate (train): {train['Churn'].mean()*100:.1f}%  <- imbalanced classes, "
      f"so accuracy alone will be misleading; precision/recall/F1 matter more")

def preprocess(df, encoders=None, fit=False):
    df = df.copy()
    df['International plan'] = (df['International plan'] == 'Yes').astype(int)
    df['Voice mail plan'] = (df['Voice mail plan'] == 'Yes').astype(int)
    df['Churn'] = df['Churn'].astype(int)

    if encoders is None:
        encoders = {}
    if fit:
        le = LabelEncoder()
        df['State'] = le.fit_transform(df['State'])
        encoders['State'] = le
    else:
        df['State'] = encoders['State'].transform(df['State'])
    return df, encoders

train_p, enc = preprocess(train, fit=True)
test_p, _ = preprocess(test, encoders=enc, fit=False)

X_train = train_p.drop(columns=['Churn'])
y_train = train_p['Churn']
X_test = test_p.drop(columns=['Churn'])
y_test = test_p['Churn']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
}

results = {}
for name, m in models.items():
    if name == 'Logistic Regression':
        m.fit(X_train_scaled, y_train)
        pred = m.predict(X_test_scaled)
    else:
        m.fit(X_train, y_train)
        pred = m.predict(X_test)
    results[name] = {
        'accuracy': accuracy_score(y_test, pred),
        'precision': precision_score(y_test, pred),
        'recall': recall_score(y_test, pred),
        'f1': f1_score(y_test, pred),
    }

results_df = pd.DataFrame(results).T.round(3)
print(f"\nModel comparison:\n{results_df}")

# Hyperparameter tuning on the best baseline model (Random Forest)
print("\nRunning grid search on Random Forest...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='f1', n_jobs=-1)
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_
pred_best = best_rf.predict(X_test)

print(f"\nBest params: {grid.best_params_}")
print(f"Tuned Random Forest -> Accuracy: {accuracy_score(y_test, pred_best):.3f}, "
      f"Precision: {precision_score(y_test, pred_best):.3f}, "
      f"Recall: {recall_score(y_test, pred_best):.3f}, "
      f"F1: {f1_score(y_test, pred_best):.3f}")

# Feature importance - the business-relevant output
importances = pd.Series(best_rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(f"\nTop 5 churn drivers (feature importance):\n{importances.head(5).round(3)}")

# Confusion matrix plot
cm = confusion_matrix(y_test, pred_best)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Stayed', 'Churned'], yticklabels=['Stayed', 'Churned'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Tuned Random Forest')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/plots/level3_confusion_matrix.png', dpi=150)
plt.close()

plt.figure(figsize=(8, 6))
importances.head(10).sort_values().plot(kind='barh', color='steelblue')
plt.title('Top 10 Churn Drivers')
plt.xlabel('Feature Importance')
plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/plots/level3_feature_importance.png', dpi=150)
plt.close()

# ---------------------------------------------------------
# TASK 2: DASHBOARD DATA PREP (for Power BI / Tableau)
# ---------------------------------------------------------
print("\n" + "=" * 60)
print("TASK 2: DASHBOARD DATA PREP")
print("=" * 60)

full_churn = pd.concat([train, test], ignore_index=True)
full_churn['Churn'] = full_churn['Churn'].map({True: 'Yes', False: 'No'})
full_churn.to_csv('/mnt/user-data/outputs/churn_dashboard_data.csv', index=False)
print("\nCombined, dashboard-ready dataset saved -> churn_dashboard_data.csv")
print("(Import this directly into Power BI / Tableau - see DASHBOARD_GUIDE.md for what to build)")
