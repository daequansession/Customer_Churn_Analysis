# Customer Churn Prediction Analysis
# By: Daequan Session

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("CUSTOMER CHURN PREDICTION ANALYSIS")
print("By: Daequan Session")
print("="*70)

print("\nLoading data...")

try:
    df = pd.read_csv('customer_churn.csv')
    print(f"Data loaded: {len(df):,} customers")
except FileNotFoundError:
    print("ERROR: customer_churn.csv not found!")
    exit()

print("\nDataset Preview:")
print(df.head())

print(f"\nDataset Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

print("\n" + "="*70)
print("DATA CLEANING")
print("="*70)

print("\nChecking for missing values...")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("No missing values found!")

print("\nChurn Distribution:")
churn_counts = df['Churn'].value_counts()
print(churn_counts)
print(f"\nChurn Rate: {(churn_counts['Yes'] / len(df) * 100):.1f}%")

print("\n" + "="*70)
print("EXPLORATORY DATA ANALYSIS")
print("="*70)

if df['TotalCharges'].dtype == 'object':
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)

print("\nANALYSIS 1: Churn by Contract Type")
contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
print(contract_churn)

print("\nANALYSIS 2: Churn by Tenure")
churned = df[df['Churn'] == 'Yes']['tenure'].mean()
stayed = df[df['Churn'] == 'No']['tenure'].mean()
print(f"Average tenure (Churned): {churned:.1f} months")
print(f"Average tenure (Stayed): {stayed:.1f} months")

print("\nANALYSIS 3: Churn by Monthly Charges")
churned_charges = df[df['Churn'] == 'Yes']['MonthlyCharges'].mean()
stayed_charges = df[df['Churn'] == 'No']['MonthlyCharges'].mean()
print(f"Average monthly charges (Churned): ${churned_charges:.2f}")
print(f"Average monthly charges (Stayed): ${stayed_charges:.2f}")

print("\n" + "="*70)
print("CREATING VISUALIZATIONS")
print("="*70)

plt.figure(figsize=(8, 6))
churn_counts.plot(kind='bar', color=['#2ecc71', '#e74c3c'])
plt.title('Customer Churn Distribution', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Churn Status', fontsize=12)
plt.ylabel('Number of Customers', fontsize=12)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('churn_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: churn_distribution.png")
plt.close()

plt.figure(figsize=(10, 6))
contract_churn_counts = pd.crosstab(df['Contract'], df['Churn'])
contract_churn_counts.plot(kind='bar', color=['#2ecc71', '#e74c3c'])
plt.title('Churn Rate by Contract Type', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Contract Type', fontsize=12)
plt.ylabel('Number of Customers', fontsize=12)
plt.legend(['Stayed', 'Churned'])
plt.xticks(rotation=45)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('churn_by_contract.png', dpi=300, bbox_inches='tight')
print("Saved: churn_by_contract.png")
plt.close()

plt.figure(figsize=(12, 6))
plt.hist([df[df['Churn']=='No']['tenure'], df[df['Churn']=='Yes']['tenure']], 
         bins=20, label=['Stayed', 'Churned'], color=['#2ecc71', '#e74c3c'])
plt.title('Customer Tenure Distribution by Churn Status', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Tenure (months)', fontsize=12)
plt.ylabel('Number of Customers', fontsize=12)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('tenure_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: tenure_distribution.png")
plt.close()

plt.figure(figsize=(10, 6))
df.boxplot(column='MonthlyCharges', by='Churn', 
           patch_artist=True, figsize=(10, 6))
plt.suptitle('')
plt.title('Monthly Charges by Churn Status', fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Churn Status', fontsize=12)
plt.ylabel('Monthly Charges ($)', fontsize=12)
plt.tight_layout()
plt.savefig('monthly_charges_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: monthly_charges_comparison.png")
plt.close()

print("\n" + "="*70)
print("BUILDING MACHINE LEARNING MODEL")
print("="*70)

print("\nPreparing data for machine learning...")

features = ['tenure', 'MonthlyCharges', 'TotalCharges']
X = df[features].copy()
y = (df['Churn'] == 'Yes').astype(int)

X = X.fillna(X.median())

print(f"Features selected: {features}")
print(f"Total samples: {len(X):,}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"Training set: {len(X_train):,} customers")
print(f"Test set: {len(X_test):,} customers")

print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("Model trained successfully!")

y_pred = model.predict(X_test)

print("\n" + "="*70)
print("MODEL PERFORMANCE")
print("="*70)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy*100:.1f}%")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['Stayed', 'Churned']))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

print("\nFeature Importance:")
feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print(feature_importance)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], 
         color='#3498db')
plt.xlabel('Importance Score', fontsize=12)
plt.title('Feature Importance for Churn Prediction', fontsize=14, fontweight='bold', pad=20)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\nSaved: feature_importance.png")
plt.close()

print("\n" + "="*70)
print("BUSINESS INSIGHTS & RECOMMENDATIONS")
print("="*70)

churn_probability = model.predict_proba(X_test)[:, 1]
at_risk_customers = (churn_probability > 0.7).sum()

print(f"\nHigh-Risk Customers Identified: {at_risk_customers}")
print(f"Percentage of Test Set: {(at_risk_customers/len(X_test)*100):.1f}%")

print("\nKEY INSIGHTS:")
print("1. Month-to-month contracts have highest churn rate")
print(f"2. Churned customers have {churned:.1f} months avg tenure vs {stayed:.1f} months")
print(f"3. Churned customers pay ${churned_charges:.2f}/month vs ${stayed_charges:.2f}/month")
print(f"4. Model can predict churn with {accuracy*100:.1f}% accuracy")

print("\nRECOMMENDATIONS:")
print("1. Offer incentives for customers to switch to long-term contracts")
print("2. Implement retention program for customers < 12 months tenure")
print("3. Review pricing strategy for high monthly charge customers")
print("4. Deploy model to score all customers monthly")

summary = {
    'Metric': [
        'Total Customers Analyzed',
        'Overall Churn Rate (%)',
        'Model Accuracy (%)',
        'High-Risk Customers Identified',
        'Avg Tenure - Churned (months)',
        'Avg Tenure - Stayed (months)',
        'Avg Monthly Charges - Churned ($)',
        'Avg Monthly Charges - Stayed ($)'
    ],
    'Value': [
        f"{len(df):,}",
        f"{(churn_counts['Yes'] / len(df) * 100):.1f}",
        f"{accuracy*100:.1f}",
        f"{at_risk_customers}",
        f"{churned:.1f}",
        f"{stayed:.1f}",
        f"{churned_charges:.2f}",
        f"{stayed_charges:.2f}"
    ]
}

summary_df = pd.DataFrame(summary)
summary_df.to_csv('churn_analysis_summary.csv', index=False)
print("\nSaved: churn_analysis_summary.csv")

print("\n" + "="*70)
print("ANALYSIS COMPLETE!")
print("="*70)

print("\nFiles Created:")
print("   1. churn_distribution.png")
print("   2. churn_by_contract.png")
print("   3. tenure_distribution.png")
print("   4. monthly_charges_comparison.png")
print("   5. feature_importance.png")
print("   6. churn_analysis_summary.csv")

