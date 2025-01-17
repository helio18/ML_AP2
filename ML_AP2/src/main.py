import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt

path_to_csv = '../data/tourism_dataset.csv'
df = pd.read_csv(path_to_csv)

print("First 5 rows of the dataset:")
print(df.head())
print("\nDataset Info:")
print(df.info())
print("\nBasic Statistics:")
print(df.describe())

df['Revenue_per_visitor'] = df['Revenue'] / df['Visitors']

df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['Revenue_per_visitor'])

cat_cols = ['Country', 'Category', 'Accommodation_Available']
label_encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df[['Country', 'Category', 'Accommodation_Available', 'Visitors', 'Rating']]

y = df['Revenue_per_visitor']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} rows")
print(f"Test set size:     {X_test.shape[0]} rows")

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

lin_mse = mean_squared_error(y_test, y_pred_lin)
lin_r2 = r2_score(y_test, y_pred_lin)

print("\n--- Linear Regression Performance ---")
print(f"MSE: {lin_mse:.4f}")
print(f"R^2: {lin_r2:.4f}")

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_r2 = r2_score(y_test, y_pred_rf)

print("\n--- Random Forest Performance ---")
print(f"MSE: {rf_mse:.4f}")
print(f"R^2: {rf_r2:.4f}")

if rf_r2 > lin_r2:
    model_to_use = rf
    best_model_name = "Random Forest"
else:
    model_to_use = lin_reg
    best_model_name = "Linear Regression"

print(f"\nChosen Model: {best_model_name}")

graphs_dir = '../data/graphs/'
if not os.path.exists(graphs_dir):
    os.makedirs(graphs_dir)
    print(f"Created directory: {graphs_dir}")

models = ["Linear Regression", "Random Forest"]
mses = [lin_mse, rf_mse]
r2s  = [lin_r2, rf_r2]

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.bar(models, mses, color=['skyblue', 'lightgreen'])
plt.title("Model Comparison: MSE")
plt.ylabel("MSE")
for i, v in enumerate(mses):
    plt.text(i, v + max(mses)*0.01, f"{v:.2f}", ha='center', va='bottom')

plt.subplot(1, 2, 2)
plt.bar(models, r2s, color=['skyblue', 'lightgreen'])
plt.title("Model Comparison: $R^2$")
plt.ylabel("$R^2$")
for i, v in enumerate(r2s):
    plt.text(i, v + max(r2s)*0.01, f"{v:.4f}", ha='center', va='bottom')

plt.tight_layout()

model_comparison_path = os.path.join(graphs_dir, 'model_comparison.png')
plt.savefig(model_comparison_path, dpi=300)
print(f"Saved model comparison plot to {model_comparison_path}")

plt.show()

desired_country_name = "France"
desired_country_label = None

country_le = label_encoders['Country']
for lbl_code in range(len(country_le.classes_)):
    if country_le.classes_[lbl_code] == desired_country_name:
        desired_country_label = lbl_code
        break

if desired_country_label is None:
    print(f"Warning: '{desired_country_name}' not found in the dataset. Using a fallback country.")
    desired_country_label = 0

num_categories = len(label_encoders['Category'].classes_)
ranked_results = []

for cat_label in range(num_categories):
    X_infer = pd.DataFrame({
        'Country': [desired_country_label],
        'Category': [cat_label],
        'Accommodation_Available': [1],
        'Visitors': [1000],
        'Rating': [3.0]
    })

    predicted_value = model_to_use.predict(X_infer)[0]
    cat_name = label_encoders['Category'].inverse_transform([cat_label])[0]
    ranked_results.append((cat_name, predicted_value))

ranked_results.sort(key=lambda x: x[1], reverse=True)

print(f"\nRanking of categories for {desired_country_name} (accommodation=Yes, rating=3, visitors=1000):")
for i, (cat_name, val) in enumerate(ranked_results, start=1):
    print(f"{i}. {cat_name} -> predicted Revenue_per_visitor = {val:.2f}")

categories = [item[0] for item in ranked_results]
scores = [item[1] for item in ranked_results]

plt.figure(figsize=(8, 6))
plt.bar(categories, scores, color='orange')
plt.title(f"Ranking for {desired_country_name} (Yes, rating=3, visitors=1000)")
plt.ylabel("Predicted Revenue per Visitor")
plt.xticks(rotation=45)

for i, v in enumerate(scores):
    plt.text(i, v + max(scores)*0.01, f"{v:.2f}", ha='center', va='bottom')

plt.tight_layout()

final_ranking_path = os.path.join(graphs_dir, 'final_ranking.png')
plt.savefig(final_ranking_path, dpi=300)
print(f"Saved final ranking plot to {final_ranking_path}")

plt.show()