# 🏠 House Price Prediction using Linear Regression

This project builds a **machine learning model** to predict **house prices** based on features such as area, number of bedrooms, bathrooms, parking, amenities, and furnishing status.

It demonstrates an end-to-end **regression workflow** using:

- **Pandas** for data handling  
- **Scikit-learn** for preprocessing and modeling  
- **OneHotEncoder + ColumnTransformer** for categorical variables  
- **Linear Regression** as a baseline model  
- **Joblib** for model persistence  

This repository is part of my **ML learning journey and portfolio**, showcasing my ability to move from a clean dataset to a trained model, evaluation, reproducible script, and documentation.

---

## 📂 Project Structure

```text
house-price-prediction-ml/
├── data/
│   └── Housing.xlsx                 # Input dataset
├── models/
│   └── house_price_model.pkl        # Trained model (generated after running script)
├── notebooks/
│   └── 01_eda_and_baseline.ipynb    # (Optional) EDA / experiment notebook
├── src/
│   └── train_model.py               # Main training & evaluation script
├── README.md                        # Project documentation (this file)
├── reflections.md                   # Personal reflection on learning
└── requirements.txt                 # Python dependencies

📊 Dataset Overview

The dataset (data/Housing.xlsx) contains information about houses and their selling prices.

Target variable:

price – house price (numeric)

Example features:

area – size of the house

bedrooms – number of bedrooms

bathrooms – number of bathrooms

stories – number of stories

mainroad – whether the house has access to a main road (yes/no)

guestroom – whether there is a guest room (yes/no)

basement – presence of a basement (yes/no)

hotwaterheating – hot water heating (yes/no)

airconditioning – air conditioning (yes/no)

parking – number of parking spaces

prefarea – preferred area or not (yes/no)

furnishingstatus – furnishing status (e.g., furnished, semi-furnished, unfurnished)

This is a tabular regression problem with a mix of numeric and categorical features.

🧠 Modeling Approach

1. Load Data
Read Housing.xlsx using pandas.read_excel.

2. Feature/Target Split
X = all feature columns
y = price

3. Preprocessing
Identify:
Categorical columns (object dtype)
Numeric columns (non-object)
Use ColumnTransformer to:
Apply OneHotEncoder to categorical columns
Pass numeric columns through unchanged

4. Model
Use LinearRegression from scikit-learn as the base model
Wrap preprocessing + model in a Pipeline

5. Train/Test Split
80% training, 20% test
random_state=42 for reproducibility

6. Evaluation
Mean Squared Error (MSE)
R² Score

7. Model Saving
Save the trained pipeline using joblib.dump to models/house_price_model.pkl

🧪 How to Run the Project
1️⃣ Create & Activate Virtual Environment (Optional but Recommended)
python -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Place the Dataset

Ensure the file structure is:
data/Housing.xlsx

4️⃣ Run the Training Script

From the project root:
python src/train_model.py

This will:
Train the model

Print evaluation metrics (MSE, R²) in the terminal
Save the trained model to: models/house_price_model.pkl

📈 Sample Output (Console)

Example of what you might see:

Mean Squared Error: <value>
R2 Score: <value>
Model saved to models/house_price_model.pkl


(Exact values will depend on the dataset and preprocessing.)
