# Practical Application III: Comparing Classifiers

## Overview

This project compares the performance of four classification algorithms—**K-Nearest Neighbors (KNN)**, **Logistic Regression**, **Decision Trees**, and **Support Vector Machines (SVM)**—on a bank marketing dataset to predict term deposit subscriptions.

**Date Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

---

## Objective

Develop a predictive classification model to identify which bank clients are most likely to subscribe to a term deposit during telemarketing campaigns, enabling the bank to:
- Optimize marketing resources
- Increase campaign success rates
- Reduce operational costs
- Improve customer experience

---

## Project Structure

```
comparing-classifiers
/
│
├── README.md                          # This file - project summary
├── prompt_III.ipynb                   # Main Jupyter Notebook with analysis
│
├── images/                            # Visualizations directory
│   ├── data_exploration.png           
│   └── model_comparison.png           
│
├── data/                              # Dataset directory
│   ├── bank-additional-full.csv       
│   ├── bank-additional.csv            
│   └── bank-additional-names.txt      
└── CRISP-DM-BANK.pdf                 
```

---

## Main Deliverable

**[Jupyter Notebook: prompt_III.ipynb](./prompt_III.ipynb)**

The notebook contains:
- Complete analysis of all 11 assignment problems
- Comparison of 4 classification algorithms
- Hyperparameter tuning and model improvement
- Visualizations and business recommendations

---

## Key Findings

### 1. **Class Imbalance Challenge**
- Dataset is severely imbalanced: **88.7% "no"** vs. **11.3% "yes"**
- Default models achieved ~88% accuracy but **0% recall** for minority class
- High accuracy was misleading—models predicted only the majority class

### 2. **Default Model Performance** (Problem 10)

| Model | Test Accuracy | Train Time | Recall (Yes) |
|-------|---------------|------------|--------------|
| **Logistic Regression** | 88.74% | 0.38s | 0.00% |
| **K-Nearest Neighbors** | 87.75% | 0.005s | 7.54% |
| **Decision Tree** | 86.33% | 0.06s | 8.51% |
| **Support Vector Machine** | 88.74% | 5.24s | 0.00% |

**Key Insight**: All models showed poor minority class detection, defeating the business purpose of identifying potential subscribers.

### 3. **Improved Model Performance** (Problem 11)

After hyperparameter tuning and class weighting optimization:

| Model | Test Accuracy | Recall (Yes) | F1-Score | Improvement |
|-------|---------------|--------------|----------|-------------|
| **Logistic Regression** | 58.47% | **62.18%** | 0.2522 | ⬆️ 62.18% recall gain |
| **K-Nearest Neighbors** | 86.26% | 9.05% | 0.1292 | ⬆️ 1.51% recall gain |
| **Decision Tree** | 87.05% | 17.56% | 0.2340 | ⬆️ 9.05% recall gain |

**Key Success Factor**: Used **F1-score** (not accuracy) as the optimization metric in GridSearchCV, forcing models to balance precision and recall.

### 4. **Overfitting Analysis**

| Model | Overfitting Gap | Status |
|-------|-----------------|--------|
| Logistic Regression | ~0.00% | ✅ Excellent |
| K-Nearest Neighbors | 1.38% | ✅ Excellent |
| Decision Tree | 5.37% | ⚠️ Moderate |
| Support Vector Machine | ~0.00% | ✅ Excellent |

Three out of four models showed excellent generalization with minimal overfitting.

### 5. **Critical Insight: The Accuracy Paradox**

**Problem**: With 88.7% negative class, a model that always predicts "no" achieves 88.74% accuracy!

**Solution**: Trade accuracy for recall
- **Before**: 88.74% accuracy, 0% recall (useless for business)
- **After**: 58.47% accuracy, 62.18% recall (catches 577 out of 928 subscribers!)

**Business Value**: Identifying 62% of potential subscribers is infinitely more valuable than identifying 0% with "high accuracy."

---

## Methodology

### Feature Engineering
- Selected 7 bank client features: age, job, marital status, education, default status, housing loan, personal loan
- Applied one-hot encoding: 7 features → 28 encoded features
- Label encoded target variable: "no" = 0, "yes" = 1

### Model Training & Evaluation
1. **Baseline**: DummyClassifier (most frequent) = 88.74% accuracy
2. **Default Models**: All 4 classifiers with default settings
3. **Hyperparameter Tuning**: GridSearchCV with 5-fold cross-validation
4. **Optimization Metric**: F1-score (critical for imbalanced data)
5. **Class Weighting**: Tested multiple strategies including 'balanced'

### Key Parameters Tuned
- **Logistic Regression**: C, penalty, solver, class_weight
- **K-Nearest Neighbors**: n_neighbors, weights, metric
- **Decision Tree**: max_depth, min_samples_split, min_samples_leaf, criterion, class_weight

---

## Recommendations

### 1. **Model Selection**
- **Recommended**: Logistic Regression with balanced class weights
- **Rationale**: Best F1-score (0.2522), highest recall (62.18%), no overfitting

### 2. **Deployment Strategy**
- Deploy model with optimized class weights
- Monitor both accuracy AND recall in production
- Use probability thresholds to adjust precision/recall trade-off

### 3. **ROI Justification**
- **Cost of wasted call**: $5-10
- **Value of new subscriber**: $100-500
- **With 62% recall and reasonable precision**: Positive ROI even with false positives
- **Catching 577 subscribers** vs. 0 with high-accuracy model

### 4. **Further Improvements**
- Include all 20 features (currently using only 7 bank features)
- Try ensemble methods (Random Forest, XGBoost, Gradient Boosting)
- Collect more positive examples for better model training
- Experiment with probability threshold tuning

---

## Visualizations

The project includes two sets of visualizations saved in the `images/` folder:

### 1. Data Exploration (`images/data_exploration.png`)
- **Class Distribution**: Bar chart showing severe class imbalance (88.7% vs 11.3%)
- **Age Distribution**: Violin plot comparing age patterns between classes
- Created with **Seaborn** for professional statistical visualization

### 2. Model Comparison (`images/model_comparison.png`)
Comprehensive 4-panel visualization created with **Matplotlib**:
1. **Test Accuracy Comparison**: All 4 models vs. baseline
2. **Training Time Comparison**: Efficiency analysis
3. **Overfitting Analysis**: Generalization assessment (color-coded)
4. **Tuned Models Metrics**: Multi-metric comparison after optimization

---

## 🛠️ Technologies Used

- **Python 3.13.5**
- **Libraries**:
  - `pandas` - Data manipulation
  - `numpy` - Numerical operations
  - `scikit-learn` - Machine learning algorithms
  - `matplotlib` - Visualizations
  
- **Models**:
  - `LogisticRegression`
  - `KNeighborsClassifier`
  - `DecisionTreeClassifier`
  - `SVC` (Support Vector Classifier)
  
- **Techniques**:
  - `train_test_split` with stratification
  - `GridSearchCV` for hyperparameter tuning
  - `LabelEncoder` and `get_dummies` for encoding
  - `DummyClassifier` for baseline

---