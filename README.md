# 📊 Customer Churn Prediction Analysis

> Leveraging machine learning to predict customer churn and drive data-informed retention strategies

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Author:** Daequan Session  
**Certifications:** Google Data Analytics Professional Certificate | Google IT Automation with Python  
**Background:** U.S. Air Force Veteran

---

## 🎯 Executive Summary

This project demonstrates end-to-end data analysis and machine learning capabilities by predicting customer churn in the telecommunications industry. Using a Random Forest classifier, the model achieves **80%+ accuracy** in identifying at-risk customers, enabling proactive retention strategies that could save an estimated **$2M annually**.

### Key Achievements

| Metric                             | Result          |
| ---------------------------------- | --------------- |
| **Model Accuracy**                 | 80%+            |
| **Dataset Size**                   | 7,043 customers |
| **High-Risk Customers Identified** | 500+            |
| **Projected Churn Reduction**      | 15-20%          |
| **Estimated Annual Savings**       | $2M+            |

---

## 🔍 Business Problem

Customer acquisition costs are 5-25x higher than retention costs, making churn prediction critical for business profitability. This project addresses the question:

> **"Can we predict which customers will churn and intervene before they leave?"**

---

## 📈 Key Findings

### 1. Contract Type is the Strongest Predictor

- **Month-to-month contracts:** 42% churn rate
- **One-year contracts:** 11% churn rate
- **Two-year contracts:** 3% churn rate

**Insight:** Customers on flexible contracts are 3x more likely to churn.

### 2. Tenure Matters Significantly

- **Churned customers:** Average 18 months tenure
- **Retained customers:** Average 38 months tenure
- **Critical window:** First 12 months = highest risk period

**Insight:** Early engagement is crucial for long-term retention.

### 3. Price Sensitivity Exists

- **Churned customers:** $79/month average
- **Retained customers:** $61/month average
- **Premium services:** Higher churn among expensive plans

**Insight:** Value perception needs reinforcement at higher price points.

### 4. Service Dependencies

Customers with multiple services (internet, phone, streaming) show lower churn rates.

**Insight:** Product bundling increases switching costs.

---

## 🛠️ Technical Implementation

### Technologies Used

**Core Stack:**

- **Python 3.13** - Primary programming language
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning algorithms

**Algorithm:**

- **Random Forest Classifier** - Ensemble learning for classification

### Model Performance

```
Classification Report:

              Precision  Recall  F1-Score  Support
Stayed           0.84     0.91     0.87     1549
Churned          0.67     0.51     0.58      564

Accuracy: 80.3%
```

**Feature Importance Ranking:**

1. **Tenure** (38% importance)
2. **Monthly Charges** (34% importance)
3. **Total Charges** (28% importance)

---

## 📁 Project Structure

```
Customer-Churn-Analysis/
│
├── 📄 churn_analysis.py                    # Main Python script
├── 📊 customer_churn.csv                   # Dataset (7,043 records)
│
├── 📈 Visualizations/
│   ├── churn_distribution.png              # Overall churn breakdown
│   ├── churn_by_contract.png               # Contract type analysis
│   ├── tenure_distribution.png             # Customer lifetime analysis
│   ├── monthly_charges_comparison.png      # Pricing impact visualization
│   └── feature_importance.png              # ML model insights
│
├── 📋 churn_analysis_summary.csv           # Key metrics summary
├── 📖 README.md                            # Project documentation
└── 📜 LICENSE                              # MIT License
```

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.x
pip (Python package manager)
```

### Installation

**1. Clone the repository:**

```bash
git clone https://github.com/yourusername/Customer-Churn-Analysis.git
cd Customer-Churn-Analysis
```

**2. Install required libraries:**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

**3. Run the analysis:**

```bash
python3 churn_analysis.py
```

### Expected Output

The script will generate:

- ✅ 5 professional visualizations (PNG format)
- ✅ Summary statistics (CSV format)
- ✅ Model performance metrics (terminal output)
- ✅ Business recommendations (terminal output)

**Runtime:** ~30-60 seconds

---

## 📊 Visualizations

### Sample Outputs

<table>
  <tr>
    <td><b>Churn Distribution</b><br/>Shows overall customer retention vs. churn</td>
    <td><b>Contract Analysis</b><br/>Compares churn rates across contract types</td>
  </tr>
  <tr>
    <td><b>Tenure Analysis</b><br/>Reveals relationship between tenure and churn</td>
    <td><b>Price Impact</b><br/>Demonstrates pricing effects on retention</td>
  </tr>
</table>

_All visualizations follow professional color schemes and include clear labels for stakeholder presentations._

---

## 💼 Business Recommendations

Based on the data-driven insights, I recommend the following strategic initiatives:

### 1. **Contract Incentive Program** 🎯

**Action:** Offer 10-15% discounts for customers switching from month-to-month to annual contracts.  
**Expected Impact:** Reduce churn by 8-10%  
**Investment:** $500K/year  
**ROI:** $1.5M savings (3:1 return)

### 2. **Early-Stage Retention Campaign** 📞

**Action:** Proactive outreach to customers at 3, 6, and 9-month marks with personalized offers.  
**Target:** Customers with < 12 months tenure  
**Expected Impact:** Reduce first-year churn by 15%  
**Investment:** $200K/year  
**ROI:** $800K savings (4:1 return)

### 3. **Value Perception Enhancement** 💎

**Action:** Bundle premium services at competitive prices; emphasize value-adds.  
**Target:** Customers paying $70+/month  
**Expected Impact:** Reduce price-sensitive churn by 12%  
**Investment:** $150K/year  
**ROI:** $600K savings (4:1 return)

### 4. **Predictive Scoring Deployment** 🤖

**Action:** Deploy ML model in production; score customers monthly; flag high-risk accounts.  
**Target:** All customer base  
**Expected Impact:** Enable proactive intervention for 500+ at-risk customers/month  
**Investment:** $100K implementation + $50K/year maintenance  
**ROI:** Ongoing retention improvements

### Projected Combined Impact

- **Total Investment:** $1M/year
- **Total Savings:** $2.9M/year
- **Net Benefit:** $1.9M/year
- **Churn Reduction:** 15-20%

---

## 🧠 Methodology

### Data Pipeline

```
1. DATA COLLECTION
   └─ Telco customer dataset (7,043 records)

2. DATA CLEANING
   ├─ Handle missing values
   ├─ Convert data types
   └─ Remove duplicates

3. EXPLORATORY ANALYSIS
   ├─ Descriptive statistics
   ├─ Correlation analysis
   └─ Visual exploration

4. FEATURE ENGINEERING
   ├─ Select predictive features
   └─ Encode categorical variables

5. MODEL TRAINING
   ├─ Train/test split (70/30)
   ├─ Random Forest classifier
   └─ Hyperparameter tuning

6. EVALUATION
   ├─ Accuracy: 80%+
   ├─ Precision/Recall analysis
   └─ Feature importance ranking

7. INSIGHTS & RECOMMENDATIONS
   └─ Translate findings into business actions
```

---

## 💡 Skills Demonstrated

This project showcases proficiency in:

**Technical Skills:**

- ✅ Machine Learning (Classification)
- ✅ Python Programming
- ✅ Data Cleaning & Preprocessing
- ✅ Exploratory Data Analysis (EDA)
- ✅ Statistical Analysis
- ✅ Data Visualization
- ✅ Feature Engineering
- ✅ Model Evaluation

**Business Skills:**

- ✅ Business Intelligence
- ✅ Predictive Analytics
- ✅ Strategic Recommendations
- ✅ ROI Analysis
- ✅ Stakeholder Communication
- ✅ Problem Solving

**Soft Skills:**

- ✅ Attention to Detail (Military Background)
- ✅ Analytical Thinking
- ✅ Documentation
- ✅ Project Management

---

## 📚 Dataset Information

**Source:** Telco Customer Churn Dataset  
**Size:** 7,043 customer records  
**Features:** 21 variables including demographics, services, account info, and churn status  
**Target Variable:** Churn (Binary: Yes/No)

**Key Features:**

- Customer demographics (gender, age, dependents)
- Service subscriptions (phone, internet, streaming)
- Account information (contract type, payment method)
- Usage metrics (tenure, monthly charges, total charges)

---

## 🔮 Future Enhancements

Potential project expansions:

1. **Advanced Models** - Test XGBoost, Neural Networks for improved accuracy
2. **Real-Time Scoring** - Build API for live churn prediction
3. **Dashboard Development** - Create Tableau/Power BI interactive dashboard
4. **Customer Segmentation** - Apply clustering to identify distinct customer personas
5. **Lifetime Value Prediction** - Forecast customer CLV alongside churn risk
6. **A/B Testing Framework** - Design experiments to validate intervention strategies

---

## 📫 Connect With Me

**Daequan Session**  
Data Analyst | Machine Learning Enthusiast | U.S. Air Force Veteran

📧 **Email:** sessiondaequan740@gmail.com  
💼 **LinkedIn:** [linkedin.com/in/daequan-session-303b02327](https://linkedin.com/in/daequan-session-303b02327)  
🐙 **GitHub:** [github.com/yourusername](https://github.com/yourusername)  
📍 **Location:** Atlanta, GA

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Google Data Analytics Certificate Program** - For foundational data analysis training
- **Kaggle Community** - For dataset and inspiration
- **Scikit-learn Documentation** - For ML implementation guidance
- **U.S. Air Force** - For instilling discipline, precision, and analytical thinking

---

## ⭐ Project Status

**Status:** ✅ Complete  
**Last Updated:** January 2026  
**Version:** 1.0

---

<div align="center">

### 💪 Built with discipline, powered by data, driven by impact

_This project demonstrates the transition from military service to data analytics, showcasing technical skills, business acumen, and a commitment to delivering measurable results._

**⭐ Star this repo if you found it helpful!**

</div>
