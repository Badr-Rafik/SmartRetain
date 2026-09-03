# SmartRetain

## Step 1: Project Overview

SmartRetain is a beginner-friendly machine learning project designed to predict bank customer churn. In simple terms, churn means a customer is likely to leave the bank or stop using its services. The project uses customer profile data and behavior data to estimate churn risk before the customer leaves.

This project is intended for a classroom or learning environment. The code is deliberately simple, readable, and well-commented so it can be explained clearly to an instructor or project reviewer.

### Business Goal
Banks want to reduce churn because losing customers can reduce revenue and increase customer-acquisition costs. SmartRetain helps the business team identify customers with high churn risk so they can take retention actions such as customer outreach, account reviews, or special offers.

### Project Scope
This project covers:
- Step 1: Overview and project documentation
- Step 2: Data cleaning
- Step 3: Exploratory Data Analysis (EDA)
- Step 4: Data preprocessing
- Step 5: Handling class imbalance
- Step 6: Feature engineering and feature selection
- Step 7: Model building
- Step 8: Model evaluation
- Step 9: Model optimization
- Step 10: Local Streamlit app (custom replacement for cloud deployment)
- Step 11: Final packaging and setup

### Included Checklist Items
- Data cleaning: missing values handled, duplicates removed, outliers clipped.
- EDA: descriptive statistics, histograms, boxplots, correlation heatmap, class distribution check.
- Preprocessing: categorical encoding, feature scaling, SMOTE imbalance handling.
- Feature engineering: engineered financial ratio features and feature selection.
- Model building: 3 classic algorithms.
- Model evaluation: accuracy, precision, recall, F1-score, ROC-AUC.
- Model optimization: GridSearchCV tuning on the best model.

### Main Files
- `data_preprocessing.py`: data loading, cleaning, EDA, and feature engineering
- `train.py`: model training, evaluation, tuning, and model saving
- `app.py`: local Streamlit interface for churn prediction
- `requirements.txt`: only the essential libraries needed to run the project

---

## Step 11: How to Run the Project

### 1. Open a Terminal
Change into the project folder:

```bash
cd "d:\machine learning\NTI\Final"
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
python train.py
```

This script will:
- load the bank churn dataset,
- clean the data,
- engineer a few explainable features,
- preprocess the data,
- train multiple models,
- compare the models with evaluation metrics,
- save the best model and preprocessing object.

### 4. Launch the Local Streamlit App
```bash
streamlit run app.py
```

Then open the browser link shown in the terminal.

---

## Data Source
The application uses the dataset `Bank_Churn.csv` located in the project folder.

---

## Expected Output
The app lets the user enter customer details such as age, credit score, balance, account activity, and salary. It then predicts whether the customer is likely to churn and shows the churn probability percentage.

