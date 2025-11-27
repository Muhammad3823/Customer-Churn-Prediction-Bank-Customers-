Objective  
Predict whether a bank customer will churn (exit) using demographic and financial features, via a Random Forest classifier.

Approach  
1. *Load Libraries & Dataset* – Pandas, NumPy, Scikit-learn, Matplotlib  
2. *Inspect Data* – Verified no missing values; selected 10 features (CreditScore to EstimatedSalary) and target (Exited)  
3. *Preprocess*  
   - Label-encoded Gender  
   - One-Hot encoded Geography (dropping first category)  
4. *Modeling*
   - Split: 80% train / 20% test (random_state=42)  
   - Trained *Random Forest Classifier* (n_estimators=200)  
5. *Evaluation & Visualization*
   - Accuracy, confusion matrix, classification report  
   - Visualized: feature importances, confusion matrix heatmap, churn distribution

Results & Insights  
- *Accuracy*: 86.75%  
- *Recall for churned customers*: 47% (model under-predicts churn)  
- Top predictive features: Age, Balance, IsActiveMember, EstimatedSalary  
- Class distribution shows imbalance (~20% churn), suggesting need for rebalancing or threshold tuning
