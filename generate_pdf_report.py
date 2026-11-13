import os
from fpdf import FPDF

class PDFReport(FPDF):
    def header(self):
        # Top banner on first page or running header
        if self.page_no() == 1:
            self.set_fill_color(30, 58, 138) # Dark Navy Blue
            self.rect(0, 0, 210, 24, 'F')
            self.set_text_color(255, 255, 255)
            self.set_font('Helvetica', 'B', 14)
            self.set_xy(15, 6)
            self.cell(180, 7, 'MAINCRAFTS TECHNOLOGY  |  AI & MACHINE LEARNING INTERNSHIP', 0, 1, 'L')
            self.set_font('Helvetica', '', 9)
            self.set_xy(15, 13)
            self.cell(180, 5, 'TASK 1: LINEAR REGRESSION MODEL REPORT (CALIFORNIA HOUSING PREDICTOR)', 0, 1, 'L')
            self.ln(10)
        else:
            self.set_text_color(100, 116, 139)
            self.set_font('Helvetica', 'I', 8)
            self.cell(120, 8, 'Task 1: California Housing Linear Regression Model Report', 0, 0, 'L')
            self.cell(60, 8, f'Page {self.page_no()} of 3', 0, 1, 'R')
            self.set_draw_color(226, 232, 240)
            self.line(15, 18, 195, 18)
            self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(148, 163, 184)
        if self.page_no() == 1:
            self.cell(0, 10, 'Page 1 of 3 | Confidential & Proprietary Report', 0, 0, 'C')
        else:
            self.cell(0, 10, f'California Housing Price Predictor | Task 1 Report', 0, 0, 'C')

def create_pdf_report():
    pdf = PDFReport(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=True, margin=18)
    assets_dir = 'd:/ML/report_assets'

    # ---------------- PAGE 1 ----------------
    pdf.add_page()
    pdf.set_margins(15, 15, 15)

    # Document Title Block
    pdf.set_font('Helvetica', 'B', 18)
    pdf.set_text_color(15, 23, 42) # Slate-900
    pdf.cell(180, 9, 'Predicting California House Values with Linear Regression', 0, 1, 'L')
    
    pdf.set_font('Helvetica', '', 9.5)
    pdf.set_text_color(71, 85, 105) # Slate-600
    pdf.cell(180, 5, 'Author: AI & ML Candidate   |   Deliverable: Task 1 Technical Report   |   Dataset: 1990 U.S. Census', 0, 1, 'L')
    pdf.ln(3)

    # Section 1: Executive Summary
    pdf.set_fill_color(241, 245, 249) # Slate-100 box
    pdf.rect(15, pdf.get_y(), 180, 24, 'F')
    pdf.set_xy(18, pdf.get_y() + 2)
    pdf.set_font('Helvetica', 'B', 10.5)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(174, 5, 'Executive Summary', 0, 1, 'L')
    pdf.set_font('Helvetica', '', 8.5)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(174, 4.2, 
        'This project implements an end-to-end Machine Learning pipeline to predict California district median house values '
        'using Ordinary Least Squares (OLS) Linear Regression. Using 20,640 census block records across 8 features, the model '
        'explains 57.6% of housing price variance (R2 = 0.5758) on unseen test data, achieving a Mean Absolute Error of $53,320. '
        'Median income proved to be the single most decisive factor driving valuations, followed by coastal proximity.')
    pdf.set_y(pdf.get_y() + 6)

    # Section 2: Dataset Characteristics
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(180, 6, '1. Dataset Architecture & Hygiene', 0, 1, 'L')
    pdf.set_font('Helvetica', '', 8.5)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(180, 4.2,
        'The dataset comprises 20,640 observations with 8 numerical predictor features and 1 target variable (MedHouseVal). '
        'Data hygiene verification confirmed zero missing entries and zero duplicated records across all 20,640 rows. '
        'The target variable represents median house value in hundreds of thousands of dollars ($100,000s).')
    pdf.ln(2)

    # Section 3: Exploratory Data Analysis (EDA)
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(180, 6, '2. Exploratory Data Analysis & Feature Dynamics', 0, 1, 'L')
    pdf.set_font('Helvetica', '', 8.5)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(180, 4.2,
        'Key EDA observations revealed two critical statistical patterns: (a) An artificial ceiling cap at 5.0 ($500,000), '
        'where ~965 high-value districts were clamped; (b) Median Income (MedInc) has an overwhelming positive Pearson '
        'correlation (r = +0.69) with home values, while other features exhibit moderate or non-linear associations.')
    pdf.ln(3)

    # Embed Figure 1 & Figure 2 side by side
    y_pos = pdf.get_y()
    target_img = os.path.join(assets_dir, 'target_dist.png')
    corr_img = os.path.join(assets_dir, 'corr_matrix.png')

    if os.path.exists(target_img):
        pdf.image(target_img, x=15, y=y_pos, w=88)
    if os.path.exists(corr_img):
        pdf.image(corr_img, x=107, y=y_pos, w=88)

    pdf.set_y(y_pos + 62)
    pdf.set_font('Helvetica', 'I', 7.5)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(90, 4, 'Figure 1.1: Target Distribution (MedHouseVal with cap at $500k)', 0, 0, 'C')
    pdf.cell(90, 4, 'Figure 1.2: Pearson Correlation Matrix across all features', 0, 1, 'C')

    # ---------------- PAGE 2 ----------------
    pdf.add_page()
    pdf.set_y(22)

    # Section 4: Spatial & Income Analysis
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(180, 6, '3. Spatial Geometry & Income Elasticity', 0, 1, 'L')
    pdf.set_font('Helvetica', '', 8.5)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(180, 4.2,
        'Spatial analysis reveals strong geographic clustering. Premium home prices (shown in yellow/green) tightly follow the '
        'Pacific coastline, with high concentrations in the San Francisco Bay Area and West Los Angeles. Conversely, Central Valley '
        'and inland regions demonstrate significantly lower median values. A regression scatter plot further confirms the '
        'direct linear trend between neighborhood median income and property values.')
    pdf.ln(3)

    y_pos2 = pdf.get_y()
    geo_img = os.path.join(assets_dir, 'geography.png')
    inc_img = os.path.join(assets_dir, 'medinc_vs_price.png')

    if os.path.exists(geo_img):
        pdf.image(geo_img, x=15, y=y_pos2, w=88)
    if os.path.exists(inc_img):
        pdf.image(inc_img, x=107, y=y_pos2, w=88)

    pdf.set_y(y_pos2 + 58)
    pdf.set_font('Helvetica', 'I', 7.5)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(90, 4, 'Figure 2.1: California Geographic Price Distribution', 0, 0, 'C')
    pdf.cell(90, 4, 'Figure 2.2: Median Income vs. House Value Regression', 0, 1, 'C')
    pdf.ln(4)

    # Section 5: Methodology & Model Architecture
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(180, 6, '4. Model Architecture & OLS Regression Equation', 0, 1, 'L')
    pdf.set_font('Helvetica', '', 8.5)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(180, 4.2,
        'The dataset was split into 80% training (16,512 samples) and 20% testing (4,128 samples) with random_state=42. '
        'An Ordinary Least Squares (OLS) Linear Regression model was fitted to minimize the residual sum of squares: '
        'min sum(y_i - y_hat_i)^2. The fitted model intercept is beta_0 = -37.0233.')
    pdf.ln(2)

    # Model Coefficients Table
    pdf.set_font('Helvetica', 'B', 8.5)
    pdf.set_fill_color(30, 58, 138)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(50, 5, 'Feature Name', 1, 0, 'L', True)
    pdf.cell(35, 5, 'OLS Weight (Beta)', 1, 0, 'R', True)
    pdf.cell(95, 5, 'Practical Interpretation', 1, 1, 'L', True)

    coef_data = [
        ('MedInc', '+0.4489', '+$44,888 per $10k increase in district median income (Primary driver)'),
        ('AveBedrms', '+0.7831', 'Positive partial slope conditional on total rooms held constant'),
        ('HouseAge', '+0.0097', '+$972 per additional year of neighborhood maturity'),
        ('Population', '-0.000002', 'Minimal direct independent effect on median price'),
        ('AveOccup', '-0.0035', 'Slight discount for higher household density / crowding'),
        ('AveRooms', '-0.1233', 'Room penalty when bedroom count and income are controlled'),
        ('Latitude', '-0.4337', 'Strong southward price gradient (Southern CA coast premium)'),
        ('Longitude', '-0.4345', 'Strong westward price gradient (Pacific coastal premium)')
    ]

    pdf.set_font('Helvetica', '', 8)
    for idx, (feat, val, desc) in enumerate(coef_data):
        bg = True if idx % 2 == 1 else False
        pdf.set_fill_color(248, 250, 252) if bg else pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(50, 4.5, feat, 1, 0, 'L', bg)
        pdf.set_font('Helvetica', 'B' if 'MedInc' in feat else '', 8)
        pdf.cell(35, 4.5, val, 1, 0, 'R', bg)
        pdf.set_font('Helvetica', '', 7.5)
        pdf.cell(95, 4.5, desc, 1, 1, 'L', bg)

    # ---------------- PAGE 3 ----------------
    pdf.add_page()
    pdf.set_y(22)

    # Section 6: Performance Evaluation
    pdf.set_font('Helvetica', 'B', 11)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(180, 6, '5. Quantitative Performance Evaluation & Diagnostics', 0, 1, 'L')
    pdf.set_font('Helvetica', '', 8.5)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(180, 4.2,
        'Model performance was benchmarked on both the training set and held-out test set to ensure generalizability '
        'and detect potential overfitting or underfitting.')
    pdf.ln(2)

    # Metrics Table
    pdf.set_font('Helvetica', 'B', 8.5)
    pdf.set_fill_color(30, 58, 138)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(45, 5, 'Evaluation Metric', 1, 0, 'L', True)
    pdf.cell(35, 5, 'Training Set (80%)', 1, 0, 'C', True)
    pdf.cell(35, 5, 'Test Set (20%)', 1, 0, 'C', True)
    pdf.cell(65, 5, 'Benchmark Assessment', 1, 1, 'L', True)

    metrics_rows = [
        ('Mean Absolute Error (MAE)', '0.5286 ($52,860)', '0.5332 ($53,320)', 'Consistent error; no train-test divergence'),
        ('Root Mean Squared Error (RMSE)', '0.7197 ($71,970)', '0.7456 ($74,560)', 'Captures sensitivity to extreme price peaks'),
        ('R2 Score (Variance Explained)', '0.6126 (61.3%)', '0.5758 (57.6%)', 'Healthy baseline; strong generalization')
    ]

    for idx, (m_name, m_tr, m_te, m_note) in enumerate(metrics_rows):
        bg = True if idx % 2 == 1 else False
        pdf.set_fill_color(248, 250, 252) if bg else pdf.set_fill_color(255, 255, 255)
        pdf.set_text_color(15, 23, 42)
        pdf.set_font('Helvetica', 'B', 8)
        pdf.cell(45, 5, m_name, 1, 0, 'L', bg)
        pdf.set_font('Helvetica', '', 8)
        pdf.cell(35, 5, m_tr, 1, 0, 'C', bg)
        pdf.cell(35, 5, m_te, 1, 0, 'C', bg)
        pdf.set_font('Helvetica', 'I', 7.5)
        pdf.cell(65, 5, m_note, 1, 1, 'L', bg)

    pdf.ln(3)

    # Diagnostic Images
    y_pos3 = pdf.get_y()
    act_img = os.path.join(assets_dir, 'actual_vs_pred.png')
    res_img = os.path.join(assets_dir, 'residuals_analysis.png')

    if os.path.exists(act_img):
        pdf.image(act_img, x=15, y=y_pos3, w=75)
    if os.path.exists(res_img):
        pdf.image(res_img, x=95, y=y_pos3, w=100)

    pdf.set_y(y_pos3 + 57)
    pdf.set_font('Helvetica', 'I', 7.5)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(80, 4, 'Figure 3.1: Actual vs. Predicted House Values', 0, 0, 'C')
    pdf.cell(100, 4, 'Figure 3.2: Residual Plot & Error Normality Distribution', 0, 1, 'C')
    pdf.ln(3)

    # Section 7: Key Limitations & Improvement Roadmap
    pdf.set_font('Helvetica', 'B', 10.5)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(180, 5, '6. Critical Diagnostic Findings & Improvement Roadmap', 0, 1, 'L')
    pdf.set_font('Helvetica', '', 8)
    pdf.set_text_color(51, 65, 85)

    roadmap_text = (
        '1. Diagnostic Residual Insights: Residuals are well-centered around zero (mean error ~0), confirming unbiased estimation. '
        'However, Figure 3.1 reveals a distinct horizontal ceiling at y = 5.0 ($500,000) where OLS consistently underpredicts capped luxury properties.\n'
        '2. Multi-Collinearity Mitigation: AveRooms and AveBedrms exhibit a correlation of 0.85. Introducing Ridge (L2) or Lasso (L1) '
        'regularization will stabilize parameter variance and eliminate sign anomalies.\n'
        '3. Feature Engineering: Engineering interaction terms (e.g., bedrooms-to-room ratio, distance to closest coast / major economic hub) '
        'will significantly enrich linear representational capacity.\n'
        '4. Non-Linear Ensemble Modeling: Housing pricing incorporates inherently non-linear geographic spatial pockets. Transitioning to '
        'Gradient Boosting (XGBoost / LightGBM) or Random Forest will capture spatial boundaries, typically boosting test R2 to > 0.83.'
    )
    pdf.multi_cell(180, 4, roadmap_text)

    # Deliverables Sign-off footer box
    pdf.ln(2)
    pdf.set_fill_color(238, 242, 255) # Indigo-50
    pdf.rect(15, pdf.get_y(), 180, 14, 'F')
    pdf.set_xy(18, pdf.get_y() + 2)
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(174, 4, 'Deliverables Verification Checklist:', 0, 1, 'L')
    pdf.set_font('Helvetica', '', 7.5)
    pdf.set_text_color(55, 65, 81)
    pdf.cell(174, 4, '[OK] Jupyter Notebook (task1_ml_linear_regression.ipynb)  |  [OK] Technical PDF Report  |  [OK] Saved Model (model.pkl) & CLI UI', 0, 1, 'L')

    report_pdf_path = 'd:/ML/Task1_California_Housing_Linear_Regression_Report.pdf'
    pdf.output(report_pdf_path)
    print(f"Report generated successfully: {report_pdf_path}")

if __name__ == '__main__':
    create_pdf_report()

