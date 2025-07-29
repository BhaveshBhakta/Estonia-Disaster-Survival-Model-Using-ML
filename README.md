## Estonia Disaster Survival Model

### Project Overview

This project aims to predict the **survival outcome of passengers on the Estonia ferry disaster** based on available demographic information. By analyzing factors such as age, sex, country of origin, and passenger category, the goal is to develop a machine learning model that can predict whether a passenger survived or perished in the maritime disaster. This historical analysis can provide insights into survival factors in extreme events.

-----

### Technical Highlights

  * **Dataset**: [Kaggle - Passenger List for the Estonia Ferry Disaster](https://www.kaggle.com/datasets/christianlillelund/passenger-list-for-the-estonia-ferry-disaster)
  * **Size**: 989 entries, 8 columns
  * **Key Features**:
      * Country, Sex, Age, Category (passenger type).
  * **Approach**:
      * Data Cleaning: Dropped 'PassengerId', 'Firstname', and 'Lastname' as they are unique identifiers and not relevant for generalization. No missing values or duplicates were found in the remaining data.
      * Exploratory Data Analysis: Histograms, Boxplots, and Heatmaps were used for visualization to understand data distributions and relationships.
      * Label Encoding: Applied to all categorical features and the target 'Survived'.
      * Handling Class Imbalance with `SMOTE` (Synthetic Minority Over-sampling Technique) on the training data. This is crucial as the original dataset is highly imbalanced (852 perished vs 137 survived).
      * Binary Classification: The target variable 'Survived' indicates survival (1) or death (0).
      * Models Used:
          * Logistic Regression, Ridge Classifier, SVC, Random Forest, XGBoost, AdaBoost, Gradient Boosting, Bagging, Decision Tree.
  * **Best Accuracy**:
      * 85.6% with XGBoost Classifier and Random Forest Classifier.
      * 84.5% with Bagging Classifier.
      * 82.1% with Decision Tree Classifier.

-----

### Purpose and Applications

  * Analyze historical data to understand **factors influencing survival in maritime disasters**.
  * Potentially inform safety regulations and emergency response protocols for passenger vessels.
  * Serve as a case study for applying machine learning to disaster data for retrospective analysis.
  * Provide insights for training and preparedness in disaster management.

-----

### Installation

Clone the repository:

```bash
git clone https://github.com/BhaveshBhakta/Estonia-Disaster-Survival-Model-Using-ML.git
cd Estonia-Disaster-Survival-Model-Using-ML
```

Install the necessary libraries:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn xgboost
```

-----

### Collaboration

We welcome contributions to improve the project. You can help by:

  * Performing comprehensive hyperparameter tuning and cross-validation for all models to achieve optimal performance.
  * Exploring more advanced feature engineering, such as creating interaction terms between existing features.
  * Investigating alternative methods for handling class imbalance beyond SMOTE.
  * Adding explainability (e.g., SHAP or LIME) to understand which demographic or category factors were most critical for survival.
