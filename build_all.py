import os
os.environ['MPLCONFIGDIR'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.matplotlib')
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)

import json
import base64
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from fpdf import FPDF

# Configure plotting style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'report_assets')
os.makedirs(assets_dir, exist_ok=True)

print("--- Step 1: Loading Dataset ---")
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'california_housing.csv')
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print("Loaded from local CSV:", df.shape)
else:
    data = fetch_california_housing(as_frame=True)
    df = pd.concat([data.data, data.target.rename('MedHouseVal')], axis=1)
    df.to_csv(csv_path, index=False)
    print("Loaded from sklearn & saved CSV:", df.shape)

# Step 2: Exploratory Data Analysis & Plots
print("--- Step 2: Performing EDA and Generating Figures ---")

# Figure 1: Target Distribution
fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
sns.histplot(df['MedHouseVal'], kde=True, color='#2563eb', bins=40, ax=ax, alpha=0.6)
ax.axvline(df['MedHouseVal'].mean(), color='#dc2626', linestyle='--', linewidth=1.5, label=f"Mean: ${df['MedHouseVal'].mean()*100:.1f}k")
ax.axvline(df['MedHouseVal'].median(), color='#16a34a', linestyle='-', linewidth=1.5, label=f"Median: ${df['MedHouseVal'].median()*100:.1f}k")
ax.set_title("Distribution of Median House Value (Target: MedHouseVal)", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Median House Value ($100,000s)", fontsize=10)
ax.set_ylabel("Frequency", fontsize=10)
ax.legend(frameon=True, facecolor='white', framealpha=0.9)
plt.tight_layout()
fig1_path = os.path.join(assets_dir, 'target_dist.png')
plt.savefig(fig1_path)
plt.close()

# Figure 2: Correlation Heatmap
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, mask=mask,
            square=True, linewidths=0.5, cbar_kws={"shrink": .8}, ax=ax)
ax.set_title("Feature Correlation Matrix (Pearson)", fontsize=13, fontweight='bold', pad=12)
plt.tight_layout()
fig2_path = os.path.join(assets_dir, 'corr_matrix.png')
plt.savefig(fig2_path)
plt.close()

# Figure 3: Geographic Distribution
fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
scatter = ax.scatter(df['Longitude'], df['Latitude'], c=df['MedHouseVal'], cmap='viridis',
                     alpha=0.4, s=df['Population']/150, edgecolors='none')
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('MedHouseVal ($100,000s)', fontsize=10)
ax.set_title("California Housing Geography (Price by Location)", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Longitude", fontsize=10)
ax.set_ylabel("Latitude", fontsize=10)
plt.tight_layout()
fig3_path = os.path.join(assets_dir, 'geography.png')
plt.savefig(fig3_path)
plt.close()

# Figure 4: MedInc vs MedHouseVal
fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
sns.regplot(data=df.sample(2000, random_state=42), x='MedInc', y='MedHouseVal',
            scatter_kws={'alpha': 0.3, 'color': '#2563eb', 's': 15},
            line_kws={'color': '#dc2626', 'linewidth': 2}, ax=ax)
ax.set_title("Median Income vs Median House Value (Sample: 2,000 points)", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Median Income ($10,000s)", fontsize=10)
ax.set_ylabel("Median House Value ($100,000s)", fontsize=10)
plt.tight_layout()
fig4_path = os.path.join(assets_dir, 'medinc_vs_price.png')
plt.savefig(fig4_path)
plt.close()

# Step 3: Train-Test Split & Model Training
print("--- Step 3: Model Training ---")
X = df.drop(columns='MedHouseVal')
y = df['MedHouseVal']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Metrics
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

print(f"Train -> MAE: {train_mae:.4f}, RMSE: {train_rmse:.4f}, R2: {train_r2:.4f}")
print(f"Test  -> MAE: {test_mae:.4f}, RMSE: {test_rmse:.4f}, R2: {test_r2:.4f}")

# Save model pickle
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')
joblib.dump(model, model_path)
print("Saved model to:", model_path)

# Figure 5: Actual vs Predicted
fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
ax.scatter(y_test, y_test_pred, alpha=0.3, color='#3b82f6', edgecolors='none', s=20)
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
ax.plot([min_val, max_val], [min_val, max_val], color='#dc2626', linestyle='--', linewidth=2, label='Perfect Prediction ($y=\hat{y}$)')
ax.set_title(f"Actual vs. Predicted House Values (Test Set, R² = {test_r2:.3f})", fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel("Actual MedHouseVal ($100k)", fontsize=10)
ax.set_ylabel("Predicted MedHouseVal ($100k)", fontsize=10)
ax.legend(loc='upper left', frameon=True)
plt.tight_layout()
fig5_path = os.path.join(assets_dir, 'actual_vs_pred.png')
plt.savefig(fig5_path)
plt.close()

# Figure 6: Residuals Analysis
residuals = y_test - y_test_pred
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)

ax1.scatter(y_test_pred, residuals, alpha=0.3, color='#6366f1', edgecolors='none', s=20)
ax1.axhline(0, color='#dc2626', linestyle='--', linewidth=1.5)
ax1.set_title("Residuals vs. Predicted Values", fontsize=11, fontweight='bold', pad=10)
ax1.set_xlabel("Predicted MedHouseVal ($100k)", fontsize=10)
ax1.set_ylabel("Residuals ($100k)", fontsize=10)

sns.histplot(residuals, kde=True, color='#059669', bins=40, ax=ax2, alpha=0.6)
ax2.axvline(0, color='#dc2626', linestyle='--', linewidth=1.5)
ax2.set_title("Residual Distribution (Error Normality)", fontsize=11, fontweight='bold', pad=10)
ax2.set_xlabel("Residual Error ($100k)", fontsize=10)
ax2.set_ylabel("Frequency", fontsize=10)

plt.tight_layout()
fig6_path = os.path.join(assets_dir, 'residuals_analysis.png')
plt.savefig(fig6_path)
plt.close()

# Figure 7: Model Coefficients
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', ascending=False)

fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
colors = ['#16a34a' if c > 0 else '#dc2626' for c in coef_df['Coefficient']]
ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors, height=0.6)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_title("Linear Regression Coefficients (Feature Weights)", fontsize=12, fontweight='bold', pad=12)
ax.set_xlabel("Coefficient Value", fontsize=10)
plt.tight_layout()
fig7_path = os.path.join(assets_dir, 'coefficients.png')
plt.savefig(fig7_path)
plt.close()

print("All figures generated successfully.")

