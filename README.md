# Problematic Internet Use and its Association with Physical Activity  
*A Machine Learning-Based Exploratory Study*
MSc Thesis – Atlantic Technological University (ATU), Galway, 2025

##  Overview
This repository contains the code and resources for my Master's Thesis:  
**"Problematic Internet Use and its Association with Physical Activity: A Machine Learning-Based Exploratory Study"** 



##  Introduction
The rapid growth of internet accessibility has transformed education, communication, and entertainment. However, excessive or uncontrolled usage can lead to **Problematic Internet Use (PIU)**, which negatively impacts academic performance, social relationships, sleep quality, and emotional well-being.  

At the same time, global trends show a **decline in physical activity** and a rise in **sedentary lifestyles**, raising concerns about adolescent digital wellbeing.  

This study investigates whether **machine learning models** can effectively predict PIU severity based on lifestyle and behavioural indicators such as **physical activity, screen time, and sleep patterns**. 

This README provides a summary of the project. A complete explanation, methodology, and detailed results can be found in the thesis report (ATU_MSc_Computing_G00473077_Thesis.pdf).



##  Problem Statement
- PIU is not yet formally recognised in diagnostic manuals such as the DSM-5, limiting systematic interventions.  
- Current studies rely heavily on small, survey-based samples and often treat PIU as a binary condition (problematic vs. non-problematic).  
- The relationship between **PIU, physical activity, and sedentary behaviour** remains underexplored, especially through **data-driven machine learning approaches**.  

This project addresses these gaps by developing predictive models that classify PIU severity into four levels: **None, Mild, Moderate, Severe**. 





##  Research Objectives
1. **Data Preparation** – Clean, preprocess, encode, and balance dataset.  
2. **Exploratory Data Analysis (EDA)** – Investigate correlations and behavioural patterns.  
3. **Model Development** – Implement supervised ML models: Decision Tree, Random Forest, XGBoost, LightGBM, CatBoost, and Stacking Ensemble.  
4. **Evaluation** – Compare models using Accuracy, F1-score, ROC-AUC, PR-AUC, and Quadratic Weighted Kappa (QWK).  
5. **Interpretability** – Analyse feature importance to identify key predictors.  
6. **Insights** – Provide recommendations for educators, parents, and policymakers.  



##  Methodology
- **Dataset** → [Child Mind Institute – Problematic Internet Use Kaggle dataset (2024)](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/).  
- **Preprocessing** → Missing value imputation, one-hot encoding, SMOTE for class balancing, and stratified train-test split.  
- **Models** →  
  - Decision Tree (baseline)  
  - Random Forest  
  - XGBoost  
  - LightGBM  
  - CatBoost  
  - Stacking Ensemble  
- **Validation** → Stratified K-Fold Cross-Validation.  
- **Hyperparameter Tuning** → Automated with Optuna.  
- **Evaluation Metrics** → Accuracy, Precision, Recall, F1-score, ROC-AUC, PR-AUC, **Quadratic Weighted Kappa (primary)**.  



##  Results
- **Random Forest** achieved the most stable results:  
  - Accuracy: 67%  
  - QWK: 0.66  
- Gradient boosting models (XGBoost, LightGBM, CatBoost) performed comparably but Random Forest offered better **robustness and interpretability**.  
- **Top Predictors** of PIU severity:  
  1. Sedentary screen time  
  2. Physical activity levels  
  3. Sleep duration  



##  Conclusion
This study demonstrates that:  
- Machine learning can reliably predict PIU severity based on behavioural and lifestyle data.  
- Random Forest was the most effective and interpretable model for ordinal PIU classification.  
- Reduced physical activity and increased sedentary screen time strongly correlate with higher PIU severity.  

The findings provide **actionable insights** for educators, parents, and policymakers to support early intervention strategies in digital wellbeing.  



##  Repository Contents
- problematic_internet_use_prediction.ipynb -> Jupyter Notebook with full code.  
- ATU_MSc_Computing_G00473077_Thesis.pdf -> Full thesis report with detailed methodology and results.  
- PIU_Code_WorkFlow_video.txt -> It has Youtube video link that explain about coding workflow.

##  Video Explanation

I've created a video that explains the project Check it out on YouTube:

https://youtu.be/eN8o2TS8qJY?si=I23ImB6dMaGfLcsW
##  Requirements

This project was implemented in **Python 3.13** using Jupyter Notebook inside VS Code.  
- Libraries: `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `xgboost`, `lightgbm`, `catboost`, `optuna`, `matplotlib`, `seaborn`  


To set up the environment, install the following dependencies:

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost lightgbm catboost optuna matplotlib seaborn jupyter

