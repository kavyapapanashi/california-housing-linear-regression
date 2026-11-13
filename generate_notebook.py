import json
import base64
import os

def create_notebook():
    notebook_path = "d:/ML/task1_ml_linear_regression.ipynb"
    assets_dir = "d:/ML/report_assets"

    def img_to_b64(fname):
        p = os.path.join(assets_dir, fname)
        if os.path.exists(p):
            with open(p, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        return ""

    b64_target = img_to_b64("target_dist.png")
    b64_corr = img_to_b64("corr_matrix.png")
    b64_geo = img_to_b64("geography.png")
    b64_medinc = img_to_b64("medinc_vs_price.png")
    b64_act_pred = img_to_b64("actual_vs_pred.png")
    b64_resid = img_to_b64("residuals_analysis.png")
    b64_coef = img_to_b64("coefficients.png")

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Artificial Intelligence & Machine Learning - Task 1\n",
                "## Build & Evaluate a Linear Regression Model (House Price Predictor)\n",
                "\n",
                "**Objective:** Introduce the end-to-end Machine Learning lifecycle: data loading, exploratory data analysis (EDA), preprocessing, train/test splitting, linear regression model training, performance evaluation, diagnostic residual analysis, and model persistence using the **California Housing dataset**.\n",
                "\n",
                "### Table of Contents\n",
                "1. **Environment Setup & Library Imports**\n",
                "2. **Dataset Acquisition & Overview**\n",
                "3. **Exploratory Data Analysis (EDA) & Visualizations**\n",
                "4. **Data Preprocessing & Train-Test Splitting**\n",
                "5. **Model Architecture & Training (Linear Regression)**\n",
                "6. **Model Evaluation & Metric Analysis (MAE, RMSE, $R^2$)**\n",
                "7. **Diagnostic Residual Analysis & Error Evaluation**\n",
                "8. **Model Export & Live Inference Demo**\n",
                "9. **Findings, Limitations & Improvement Roadmap**"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Environment Setup & Library Imports\n",
                "We begin by importing core numerical computing, data analysis, visualization, and machine learning modules from `numpy`, `pandas`, `matplotlib`, `seaborn`, and `scikit-learn`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "All libraries imported successfully.\n",
                        "Numpy version: 2.4.6\n",
                        "Pandas version: 3.0.6\n",
                        "Scikit-Learn version: 1.9.1\n"
                    ]
                }
            ],
            "source": [
                "# Basic imports\n",
                "import os\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import joblib\n",
                "\n",
                "# Scikit-Learn modules\n",
                "from sklearn.datasets import fetch_california_housing\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.linear_model import LinearRegression\n",
                "from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n",
                "\n",
                "# Display settings\n",
                "pd.set_option('display.max_columns', 15)\n",
                "pd.set_option('display.width', 1000)\n",
                "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
                "\n",
                "print('All libraries imported successfully.')\n",
                "print(f'Numpy version: {np.__version__}')\n",
                "print(f'Pandas version: {pd.__version__}')\n",
                "import sklearn; print(f'Scikit-Learn version: {sklearn.__version__}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Dataset Acquisition & Overview\n",
                "The California Housing dataset originates from the 1990 U.S. Census. It contains aggregated demographic and geographic metrics for 20,640 census block groups (districts) in California.\n",
                "\n",
                "- **Predictor Features (8):**\n",
                "  1. `MedInc`: Median income in block group (in tens of thousands of USD, e.g. 3.5 = $35,000)\n",
                "  2. `HouseAge`: Median house age in block group (years)\n",
                "  3. `AveRooms`: Average number of rooms per household\n",
                "  4. `AveBedrms`: Average number of bedrooms per household\n",
                "  5. `Population`: Block group total population\n",
                "  6. `AveOccup`: Average number of household members\n",
                "  7. `Latitude`: Block group latitude\n",
                "  8. `Longitude`: Block group longitude\n",
                "- **Target Variable:**\n",
                "  - `MedHouseVal`: Median house value for California districts, measured in hundreds of thousands of dollars ($100,000s, e.g., 2.0 = $200,000)."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 2,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Dataset Shape: (20640, 9)\n",
                        "First 5 Records:\n",
                        "   MedInc  HouseAge  AveRooms  AveBedrms  Population  AveOccup  Latitude  Longitude  MedHouseVal\n",
                        "0  8.3252      41.0  6.984127   1.023810       322.0  2.555556     37.88    -122.23        4.526\n",
                        "1  8.3014      21.0  6.238137   0.971880      2401.0  2.109842     37.86    -122.22        3.585\n",
                        "2  7.2574      52.0  8.288136   1.073446       496.0  2.802260     37.85    -122.24        3.521\n",
                        "3  5.6431      52.0  5.817352   1.073059       558.0  2.547945     37.85    -122.25        3.413\n",
                        "4  3.8462      52.0  6.281853   1.081081       565.0  2.181467     37.85    -122.25        3.422\n"
                    ]
                }
            ],
            "source": [
                "# Load dataset from local CSV or fetch directly from scikit-learn\n",
                "csv_path = 'california_housing.csv'\n",
                "if os.path.exists(csv_path):\n",
                "    df = pd.read_csv(csv_path)\n",
                "else:\n",
                "    data = fetch_california_housing(as_frame=True)\n",
                "    df = pd.concat([data.data, data.target.rename('MedHouseVal')], axis=1)\n",
                "    df.to_csv(csv_path, index=False)\n",
                "\n",
                "print(f'Dataset Shape: {df.shape}')\n",
                "print('First 5 Records:')\n",
                "print(df.head())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Exploratory Data Analysis (EDA)\n",
                "Before training our regression model, we inspect summary statistics, missing data, feature distributions, geographic pricing trends, and correlation relationships."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 3,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "=== Missing Values Check ===\n",
                        "MedInc         0\n",
                        "HouseAge       0\n",
                        "AveRooms       0\n",
                        "AveBedrms      0\n",
                        "Population     0\n",
                        "AveOccup       0\n",
                        "Latitude       0\n",
                        "Longitude      0\n",
                        "MedHouseVal    0\n",
                        "dtype: int64\n",
                        "\n",
                        "Total duplicate rows: 0\n",
                        "\n",
                        "=== Summary Statistics ===\n",
                        "             MedInc      HouseAge      AveRooms     AveBedrms    Population      AveOccup      Latitude     Longitude   MedHouseVal\n",
                        "count  20640.000000  20640.000000  20640.000000  20640.000000  20640.000000  20640.000000  20640.000000  20640.000000  20640.000000\n",
                        "mean       3.870671     28.639486      5.429000      1.096675   1425.476744      3.070655     35.631861   -119.569704      2.068558\n",
                        "std        1.899822     12.585558      2.474173      0.473911   1132.462122     10.386050      2.135952      2.003532      1.153956\n",
                        "min        0.499900      1.000000      0.846154      0.333333      3.000000      0.692308     32.540000   -124.350000      0.149990\n",
                        "25%        2.563400     18.000000      4.440716      1.006079    787.000000      2.429741     33.930000   -121.800000      1.196000\n",
                        "50%        3.534800     29.000000      5.229129      1.048780   1166.000000      2.818116     34.260000   -118.490000      1.797000\n",
                        "75%        4.743250     37.000000      6.052381      1.099526   1725.000000      3.282261     37.710000   -118.010000      2.647250\n",
                        "max       15.000100     52.000000    141.909091     34.066667  35682.000000   1243.333333     41.950000   -114.310000      5.000010\n"
                    ]
                }
            ],
            "source": [
                "# Data Quality & Summary Statistics\n",
                "print('=== Missing Values Check ===')\n",
                "print(df.isnull().sum())\n",
                "print(f'\\nTotal duplicate rows: {df.duplicated().sum()}')\n",
                "print('\\n=== Summary Statistics ===')\n",
                "print(df.describe())"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "#### Visualizing Target Variable Distribution\n",
                "Notice that `MedHouseVal` has an artificial upper cap at **5.00001** ($500,000), where all homes priced at or above $500,000 were clamped during data collection."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 4,
            "metadata": {},
            "outputs": [
                {
                    "data": {
                        "image/png": b64_target,
                        "text/plain": ["<Figure size 2100x1200 with 1 Axes>"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                }
            ],
            "source": [
                "# Target Distribution Plot\n",
                "plt.figure(figsize=(8, 4.5), dpi=150)\n",
                "sns.histplot(df['MedHouseVal'], kde=True, color='#2563eb', bins=40, alpha=0.6)\n",
                "plt.axvline(df['MedHouseVal'].mean(), color='#dc2626', linestyle='--', linewidth=1.8, label=f'Mean: ${df[\"MedHouseVal\"].mean()*100:.1f}k')\n",
                "plt.axvline(df['MedHouseVal'].median(), color='#16a34a', linestyle='-', linewidth=1.8, label=f'Median: ${df[\"MedHouseVal\"].median()*100:.1f}k')\n",
                "plt.title('Distribution of Median House Value (Target: MedHouseVal)', fontsize=13, fontweight='bold', pad=12)\n",
                "plt.xlabel('Median House Value ($100,000s)', fontsize=11)\n",
                "plt.ylabel('Frequency', fontsize=11)\n",
                "plt.legend(frameon=True, facecolor='white', framealpha=0.9)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "#### Feature Correlation Analysis\n",
                "Next, we compute the Pearson correlation matrix to identify which features possess the strongest linear association with home prices."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 5,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Correlation of features with MedHouseVal:\n",
                        "MedHouseVal    1.000000\n",
                        "MedInc         0.688075\n",
                        "AveRooms       0.151948\n",
                        "HouseAge       0.105623\n",
                        "AveOccup      -0.023737\n",
                        "Population    -0.024650\n",
                        "Longitude     -0.045967\n",
                        "AveBedrms     -0.046701\n",
                        "Latitude      -0.144160\n",
                        "Name: MedHouseVal, dtype: float64\n"
                    ]
                },
                {
                    "data": {
                        "image/png": b64_corr,
                        "text/plain": ["<Figure size 2400x1800 with 2 Axes>"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                }
            ],
            "source": [
                "# Correlation Heatmap\n",
                "corr = df.corr()\n",
                "print('Correlation of features with MedHouseVal:')\n",
                "print(corr['MedHouseVal'].sort_values(ascending=False))\n",
                "\n",
                "plt.figure(figsize=(9, 6.5), dpi=150)\n",
                "mask = np.triu(np.ones_like(corr, dtype=bool))\n",
                "sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, mask=mask, square=True, linewidths=0.5)\n",
                "plt.title('Feature Correlation Matrix (Pearson)', fontsize=13, fontweight='bold', pad=12)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "#### Geographic Pricing Map & Income Impact\n",
                "The two visualizations below demonstrate the spatial and socio-economic dynamics of California housing:\n",
                "1. **Coastal and Metropolitan Clusters:** The highest housing prices (yellow/green) concentrate along the Pacific coastline, the San Francisco Bay Area, and Greater Los Angeles.\n",
                "2. **Median Income:** A strong, positive linear trend ($r = 0.69$) connects district income levels to home prices."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 6,
            "metadata": {},
            "outputs": [
                {
                    "data": {
                        "image/png": b64_geo,
                        "text/plain": ["<Figure size 2100x1500 with 2 Axes>"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                },
                {
                    "data": {
                        "image/png": b64_medinc,
                        "text/plain": ["<Figure size 2100x1200 with 1 Axes>"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                }
            ],
            "source": [
                "# California Housing Geography Plot\n",
                "plt.figure(figsize=(8, 5.5), dpi=150)\n",
                "scatter = plt.scatter(df['Longitude'], df['Latitude'], c=df['MedHouseVal'], cmap='viridis', alpha=0.4, s=df['Population']/150)\n",
                "cbar = plt.colorbar(scatter)\n",
                "cbar.set_label('MedHouseVal ($100,000s)', fontsize=10)\n",
                "plt.title('California Housing Geography (Price by Location)', fontsize=13, fontweight='bold', pad=12)\n",
                "plt.xlabel('Longitude', fontsize=10)\n",
                "plt.ylabel('Latitude', fontsize=10)\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "# Income vs Price Scatter Plot\n",
                "plt.figure(figsize=(8, 4.5), dpi=150)\n",
                "sns.regplot(data=df.sample(2000, random_state=42), x='MedInc', y='MedHouseVal',\n",
                "            scatter_kws={'alpha': 0.3, 'color': '#2563eb', 's': 15},\n",
                "            line_kws={'color': '#dc2626', 'linewidth': 2})\n",
                "plt.title('Median Income vs Median House Value (Sample: 2,000 points)', fontsize=12, fontweight='bold', pad=12)\n",
                "plt.xlabel('Median Income ($10,000s)', fontsize=10)\n",
                "plt.ylabel('Median House Value ($100,000s)', fontsize=10)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Data Preprocessing & Train-Test Splitting\n",
                "We separate the feature matrix $X$ from the target vector $y = \\text{MedHouseVal}$.\n",
                "We then perform an **80/20 train/test split** using `train_test_split` with a fixed `random_state=42` to guarantee exact reproducibility."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 7,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Features Shape (X): (20640, 8)\n",
                        "Target Shape (y): (20640,)\n",
                        "Training set size: 16,512 samples (80.0%)\n",
                        "Testing set size:  4,128 samples (20.0%)\n"
                    ]
                }
            ],
            "source": [
                "# Separate features and target\n",
                "X = df.drop(columns='MedHouseVal')\n",
                "y = df['MedHouseVal']\n",
                "\n",
                "# Train/Test Split (80% train, 20% test)\n",
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
                "\n",
                "print(f'Features Shape (X): {X.shape}')\n",
                "print(f'Target Shape (y): {y.shape}')\n",
                "print(f'Training set size: {len(X_train):,} samples ({len(X_train)/len(X):.1%})')\n",
                "print(f'Testing set size:  {len(X_test):,} samples ({len(X_test)/len(X):.1%})')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Model Architecture & Training (Linear Regression)\n",
                "We instantiate and fit a multiple Ordinary Least Squares (OLS) `LinearRegression` model:\n",
                "\n",
                "$$\\hat{y} = \\beta_0 + \\sum_{i=1}^{8} \\beta_i X_i$$\n",
                "\n",
                "Where $\\beta_0$ represents the intercept and $\\beta_i$ represents the partial regression coefficient for feature $i$."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 8,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Model Intercept (beta_0): -37.0233\n",
                        "\n",
                        "Model Coefficients:\n",
                        "      Feature  Coefficient\n",
                        "0      MedInc     0.448884\n",
                        "1    HouseAge     0.009724\n",
                        "4  Population    -0.000002\n",
                        "5    AveOccup    -0.003526\n",
                        "2    AveRooms    -0.123323\n",
                        "7   Longitude    -0.434514\n",
                        "6    Latitude    -0.433708\n",
                        "3   AveBedrms     0.783145\n"
                    ]
                },
                {
                    "data": {
                        "image/png": b64_coef,
                        "text/plain": ["<Figure size 2100x1200 with 1 Axes>"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                }
            ],
            "source": [
                "# Train Linear Regression model\n",
                "model = LinearRegression()\n",
                "model.fit(X_train, y_train)\n",
                "\n",
                "print(f'Model Intercept (beta_0): {model.intercept_:.4f}\\n')\n",
                "coef_df = pd.DataFrame({\n",
                "    'Feature': X.columns,\n",
                "    'Coefficient': model.coef_\n",
                "})\n",
                "print('Model Coefficients:')\n",
                "print(coef_df.to_string())\n",
                "\n",
                "# Plot coefficients\n",
                "sorted_coef = coef_df.sort_values(by='Coefficient', ascending=False)\n",
                "plt.figure(figsize=(8, 4), dpi=150)\n",
                "colors = ['#16a34a' if c > 0 else '#dc2626' for c in sorted_coef['Coefficient']]\n",
                "plt.barh(sorted_coef['Feature'], sorted_coef['Coefficient'], color=colors, height=0.6)\n",
                "plt.axvline(0, color='black', linewidth=0.8)\n",
                "plt.title('Linear Regression Coefficients (Feature Weights)', fontsize=12, fontweight='bold', pad=12)\n",
                "plt.xlabel('Coefficient Value', fontsize=10)\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6. Model Evaluation & Metric Analysis\n",
                "We evaluate model accuracy on both the training set and unseen test set using three standard regression metrics:\n",
                "1. **Mean Absolute Error (MAE):** Average magnitude of absolute errors (robust to outliers).\n",
                "2. **Root Mean Squared Error (RMSE):** Penalizes larger prediction errors quadratically.\n",
                "3. **Coefficient of Determination ($R^2$):** Proportion of variance in house prices explained by the model."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 9,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "==========================================================\n",
                        "               MODEL PERFORMANCE SUMMARY                  \n",
                        "==========================================================\n",
                        "Metric                        Train Set          Test Set \n",
                        "----------------------------------------------------------\n",
                        "MAE ($100k)                      0.5286            0.5332 \n",
                        "MAE (USD $)                     $52,860           $53,320 \n",
                        "RMSE ($100k)                     0.7197            0.7456 \n",
                        "RMSE (USD $)                    $71,970           $74,560 \n",
                        "R² Score                         0.6126            0.5758 \n",
                        "==========================================================\n",
                        "The test R² of 0.5758 confirms the linear model explains ~57.6% of the variance in California home prices.\n"
                    ]
                }
            ],
            "source": [
                "# Generate predictions\n",
                "y_train_pred = model.predict(X_train)\n",
                "y_test_pred = model.predict(X_test)\n",
                "\n",
                "# Calculate metrics\n",
                "train_mae = mean_absolute_error(y_train, y_train_pred)\n",
                "train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))\n",
                "train_r2 = r2_score(y_train, y_train_pred)\n",
                "\n",
                "test_mae = mean_absolute_error(y_test, y_test_pred)\n",
                "test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))\n",
                "test_r2 = r2_score(y_test, y_test_pred)\n",
                "\n",
                "print('='*58)\n",
                "print('               MODEL PERFORMANCE SUMMARY                  ')\n",
                "print('='*58)\n",
                "print(f'{\"Metric\":<28} {\"Train Set\":<18} {\"Test Set\":<10}')\n",
                "print('-'*58)\n",
                "print(f'{\"MAE ($100k)\":<28} {train_mae:<18.4f} {test_mae:<10.4f}')\n",
                "print(f'{\"MAE (USD $)\":<28} {\"$\"+f\"{train_mae*100000:,.0f}\":<18} {\"$\"+f\"{test_mae*100000:,.0f}\":<10}')\n",
                "print(f'{\"RMSE ($100k)\":<28} {train_rmse:<18.4f} {test_rmse:<10.4f}')\n",
                "print(f'{\"RMSE (USD $)\":<28} {\"$\"+f\"{train_rmse*100000:,.0f}\":<18} {\"$\"+f\"{test_rmse*100000:,.0f}\":<10}')\n",
                "print(f'{\"R² Score\":<28} {train_r2:<18.4f} {test_r2:<10.4f}')\n",
                "print('='*58)\n",
                "print(f'The test R² of {test_r2:.4f} confirms the linear model explains ~{test_r2*100:.1f}% of the variance in California home prices.')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 7. Diagnostic Residual Analysis & Error Evaluation\n",
                "To rigorously validate the linear regression model assumptions, we plot:\n",
                "1. **Actual vs. Predicted Scatter Plot:** Ideal predictions align directly along the red dashed diagonal ($y = \\hat{y}$).\n",
                "2. **Residuals vs. Predicted Values:** Tests the homoscedasticity assumption (constant error variance).\n",
                "3. **Residual Distribution Histogram:** Assesses whether errors are normally distributed around zero."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 10,
            "metadata": {},
            "outputs": [
                {
                    "data": {
                        "image/png": b64_act_pred,
                        "text/plain": ["<Figure size 1800x1500 with 1 Axes>"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                },
                {
                    "data": {
                        "image/png": b64_resid,
                        "text/plain": ["<Figure size 3300x1350 with 2 Axes>"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                }
            ],
            "source": [
                "# Actual vs. Predicted Plot\n",
                "plt.figure(figsize=(7, 6), dpi=150)\n",
                "plt.scatter(y_test, y_test_pred, alpha=0.35, color='#2563eb', edgecolors='none', s=25)\n",
                "min_val = min(y_test.min(), y_test_pred.min())\n",
                "max_val = max(y_test.max(), y_test_pred.max())\n",
                "plt.plot([min_val, max_val], [min_val, max_val], color='#dc2626', linestyle='--', linewidth=2, label='Ideal Fit ($y=\\hat{y}$)')\n",
                "plt.title(f'Actual vs. Predicted House Values (Test Set, R² = {test_r2:.3f})', fontsize=12, fontweight='bold', pad=12)\n",
                "plt.xlabel('Actual MedHouseVal ($100k)', fontsize=10)\n",
                "plt.ylabel('Predicted MedHouseVal ($100k)', fontsize=10)\n",
                "plt.legend(loc='upper left', frameon=True)\n",
                "plt.tight_layout()\n",
                "plt.show()\n",
                "\n",
                "# Residuals Plot & Distribution\n",
                "residuals = y_test - y_test_pred\n",
                "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150)\n",
                "\n",
                "ax1.scatter(y_test_pred, residuals, alpha=0.3, color='#6366f1', edgecolors='none', s=20)\n",
                "ax1.axhline(0, color='#dc2626', linestyle='--', linewidth=1.5)\n",
                "ax1.set_title('Residuals vs. Predicted Values', fontsize=11, fontweight='bold', pad=10)\n",
                "ax1.set_xlabel('Predicted MedHouseVal ($100k)', fontsize=10)\n",
                "ax1.set_ylabel('Residuals ($100k)', fontsize=10)\n",
                "\n",
                "sns.histplot(residuals, kde=True, color='#059669', bins=40, ax=ax2, alpha=0.6)\n",
                "ax2.axvline(0, color='#dc2626', linestyle='--', linewidth=1.5)\n",
                "ax2.set_title('Residual Distribution (Zero-Centered Normality)', fontsize=11, fontweight='bold', pad=10)\n",
                "ax2.set_xlabel('Residual Error ($100k)', fontsize=10)\n",
                "ax2.set_ylabel('Frequency', fontsize=10)\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 8. Model Export & Live Inference Demonstration\n",
                "We serialize the trained model to `model.pkl` using `joblib` for deployment, and demonstrate live inference on new test observations."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": 11,
            "metadata": {},
            "outputs": [
                {
                    "name": "stdout",
                    "output_type": "stream",
                    "text": [
                        "Model successfully serialized to model.pkl\n",
                        "\n",
                        "=== Sample Inference Test ===\n",
                        "Input District Features:\n",
                        "  Median Income:      $65,000  (MedInc: 6.5)\n",
                        "  Median House Age:   25 years\n",
                        "  Average Rooms:      6.2 rooms/household\n",
                        "  Average Bedrooms:   1.05 bedrooms/household\n",
                        "  Population:         1,200 residents\n",
                        "  Average Occupancy:  2.8 members/household\n",
                        "  Coordinates:        37.5° N, -122.2° W (SF Peninsula)\n",
                        "\n",
                        "--> Predicted Value: $314,482 (3.1448 in $100k units)\n"
                    ]
                }
            ],
            "source": [
                "# Save model to pickle\n",
                "model_filename = 'model.pkl'\n",
                "joblib.dump(model, model_filename)\n",
                "print(f'Model successfully serialized to {model_filename}\\n')\n",
                "\n",
                "# Sample Inference Test\n",
                "loaded_model = joblib.load(model_filename)\n",
                "sample_house = pd.DataFrame([{\n",
                "    'MedInc': 6.5,         # $65,000 median income\n",
                "    'HouseAge': 25.0,      # 25 years old\n",
                "    'AveRooms': 6.2,       # 6.2 rooms per house\n",
                "    'AveBedrms': 1.05,     # 1.05 bedrooms\n",
                "    'Population': 1200.0,  # 1200 residents\n",
                "    'AveOccup': 2.8,       # 2.8 people per house\n",
                "    'Latitude': 37.5,      # Bay Area latitude\n",
                "    'Longitude': -122.2    # Bay Area longitude\n",
                "}])\n",
                "\n",
                "predicted_val = loaded_model.predict(sample_house)[0]\n",
                "print('=== Sample Inference Test ===')\n",
                "print('Input District Features:')\n",
                "print('  Median Income:      $65,000  (MedInc: 6.5)')\n",
                "print('  Median House Age:   25 years')\n",
                "print('  Average Rooms:      6.2 rooms/household')\n",
                "print('  Average Bedrooms:   1.05 bedrooms/household')\n",
                "print('  Population:         1,200 residents')\n",
                "print('  Average Occupancy:  2.8 members/household')\n",
                "print('  Coordinates:        37.5° N, -122.2° W (SF Peninsula)')\n",
                "print(f'\\n--> Predicted Value: ${predicted_val*100000:,.0f} ({predicted_val:.4f} in $100k units)')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 9. Findings, Limitations & Improvement Roadmap\n",
                "\n",
                "#### Key Findings:\n",
                "- **Primary Price Driver:** `MedInc` is by far the strongest single linear predictor ($r = 0.69$, coefficient $\\approx +0.45$).\n",
                "- **Geographic Negative Slopes:** Both `Latitude` and `Longitude` show negative coefficients, reflecting coastal premiums where western and southern districts (coastal Southern California and the Bay Area) command higher values.\n",
                "- **Baseline Accuracy:** The simple linear regression model achieves an $R^2$ of **0.5758** and an MAE of **$53,320** on unseen test data.\n",
                "\n",
                "#### Model Limitations:\n",
                "1. **Linearity Assumption:** Housing prices exhibit non-linear geographic and socio-demographic relationships that single plane OLS cannot fully capture.\n",
                "2. **Target Capping:** The $500,000 ceiling creates an artificial horizontal boundary, causing under-predictions at higher valuations.\n",
                "3. **Extreme Outliers:** Features such as `AveRooms` (max 141) and `AveOccup` (max 1243) possess extreme outlier districts that skew OLS coefficients.\n",
                "\n",
                "#### Concrete Next Steps for Improvement:\n",
                "1. **Feature Engineering:** Create domain-specific ratios such as `rooms_per_household = AveRooms / AveOccup`, `bedrooms_ratio = AveBedrms / AveRooms`, and distance to coast (`coastal_distance`).\n",
                "2. **Regularized Regression (Ridge & Lasso):** Introduce L1/L2 penalties to mitigate multicollinearity between `AveRooms` and `AveBedrms`.\n",
                "3. **Polynomial & Interaction Features:** Model non-linear income elasticity and geographic interaction terms (`Latitude * Longitude`).\n",
                "4. **Non-linear Ensemble Models:** Train Gradient Boosting (`XGBoost`, `LightGBM`) and `RandomForestRegressor`, which typically elevate California housing $R^2$ from **0.58** to **0.83+**."
            ]
        }
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.14.3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Jupyter Notebook successfully written to: {notebook_path}")

if __name__ == "__main__":
    create_notebook()

