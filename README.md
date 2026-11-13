# Maincrafts Technology - AI & Machine Learning Task 1
## Build & Evaluate a Linear Regression Model (House Price Predictor)

### 📌 Project Overview
This project completes all requirements for **Task 1** of the Artificial Intelligence & Machine Learning program. It introduces an end-to-end ML workflow: data loading, exploratory data analysis (EDA), data preprocessing, train/test splitting, linear regression model training, diagnostic residual analysis, model serialization, and technical reporting.

---

### 📂 Deliverables Summary

| Deliverable | File Path | Description |
| :--- | :--- | :--- |
| **Jupyter Notebook** | [`task1_ml_linear_regression.ipynb`](./task1_ml_linear_regression.ipynb) | Complete step-by-step notebook with code, markdown commentary, execution outputs, and embedded visualizations. |
| **PDF Summary Report** | [`Task1_California_Housing_Linear_Regression_Report.pdf`](./Task1_California_Housing_Linear_Regression_Report.pdf) | A 3-page publication-quality technical report summarizing EDA, methodology, performance metrics, and improvement recommendations. |
| **Trained Model** | [`model.pkl`](./model.pkl) | Serialized `LinearRegression` model ready for deployment or inference. |
| **Interactive CLI UI** | [`predict.py`](./predict.py) | Interactive script to run real-time price predictions on custom inputs or regional presets. |
| **Dataset (Local Copy)** | [`california_housing.csv`](./california_housing.csv) | Offline CSV copy of the California Housing dataset (20,640 records, 9 columns). |

---

### 📊 Model Performance Highlights

| Metric | Training Set (80%) | Test Set (20%) | Practical Meaning |
| :--- | :---: | :---: | :--- |
| **Mean Absolute Error (MAE)** | 0.5286 ($52,860) | **0.5332 ($53,320)** | Average dollar error is ~$53.3k across all price ranges. |
| **Root Mean Squared Error (RMSE)** | 0.7197 ($71,970) | **0.7456 ($74,560)** | Penalizes large errors, sensitive to extreme property values. |
| **Coefficient of Determination ($R^2$)** | 0.6126 (61.3%) | **0.5758 (57.6%)** | The linear model explains **57.6% of price variance** on unseen test data. |

---

### 🚀 How to Run & Test

#### 1. Launch the Prediction CLI
You can test predictions interactively or run predefined regional presets:

```bash
# Preset 1: San Francisco Bay Area Luxury District
python predict.py --preset 1

# Preset 3: Inland Central Valley (Affordable Rural District)
python predict.py --preset 3

# Interactive menu (custom inputs)
python predict.py
```

#### 2. Open the Jupyter Notebook
Open [`task1_ml_linear_regression.ipynb`](./task1_ml_linear_regression.ipynb) in **VS Code**, **Jupyter Notebook**, or **JupyterLab**. All code cells and plots are pre-rendered and ready for review.

```bash
jupyter notebook task1_ml_linear_regression.ipynb
```

#### 3. Regenerate Artifacts (Optional)
If you ever need to rebuild all plots, the notebook, or the PDF report from scratch:
```bash
python build_all.py
python generate_notebook.py
python generate_pdf_report.py
```

---

### 📈 Key Insights & Improvement Roadmap
1. **Primary Driver:** District Median Income (`MedInc`) is the strongest linear predictor ($r = 0.69$, weight $\beta \approx +0.45$).
2. **Coastal Premium:** Housing values exhibit steep spatial gradients ($Latitude$ and $Longitude$ both carry negative weights), with coastal Southern California and the SF Bay Area commanding significant premiums.
3. **Data Ceiling:** The $500,000 artificial cap in the 1990 census leads to systematic underprediction at the luxury tier.
4. **Roadmap to 80%+ $R^2$:**
   - **Feature Engineering:** Calculate bedrooms-per-room ratio (`AveBedrms / AveRooms`) and distance to the coast.
   - **Regularization:** Apply Ridge/Lasso to stabilize multicollinearity between `AveRooms` and `AveBedrms`.
   - **Non-Linear Tree Ensembles:** Deploying `RandomForestRegressor` or `XGBoostRegressor` typically boosts $R^2$ from **0.58 to 0.83+**.

