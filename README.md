# diabetes_prediction_app

1. Data Loading and Exploratory Data Analysis (EDA):

The diabetes.csv dataset was loaded and initial EDA was performed. The dataset contains 9 features and 768 entries.
A significant observation was the presence of '0' values in columns like 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', and 'BMI', which were identified as missing data rather than actual zero measurements.
No explicit null values or duplicate rows were found using .isnull().sum() and .duplicated().sum().
Histograms revealed the distributions of numerical features, showing skewness and the impact of the '0' values. The correlation heatmap highlighted strong positive correlations of 'Glucose', 'BMI', and 'Age' with the 'Outcome' variable.

2. Data Preprocessing:

Missing '0' values in key health-related features ('Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI') were imputed using the median of their respective columns to maintain data integrity and prevent distortion from zeros.
Features (X) and the target variable (y, 'Outcome') were separated.
StandardScaler was applied to X to normalize the features, which is essential for algorithms sensitive to feature scaling, ensuring fair contribution from all features during model training.

3. Model Building:

The preprocessed data was split into training (75%) and testing (25%) sets using train_test_split with random_state=42 for reproducibility.
A Logistic Regression model was initialized with max_iter=200 and random_state=42 and successfully trained on the X_train and y_train data.

4. Model Performance Evaluation:

The model's performance on the test set was evaluated using several metrics:
Accuracy: 0.7344
Precision: 0.6364
Recall: 0.6087
F1-Score: 0.6222
ROC-AUC Score: 0.7953
A Receiver Operating Characteristic (ROC) curve was plotted, visually confirming the model's ability to discriminate between diabetic and non-diabetic cases, with an Area Under the Curve (AUC) of approximately 0.80.

5. Interpretation of Model Coefficients:

The coefficients of the logistic regression model were extracted to understand the influence of each feature on the prediction.
The most influential features, sorted by the absolute value of their coefficients, were:
Glucose: With the highest positive coefficient (1.219), indicating that higher glucose levels significantly increase the likelihood of diabetes.
BMI: Also a strong positive predictor (0.706), suggesting that higher BMI is associated with a greater risk of diabetes.
Age: A notable positive coefficient (0.396), implying that older individuals are more prone to diabetes.
Other features like 'Pregnancies', 'Insulin', 'BloodPressure', 'DiabetesPedigreeFunction', and 'SkinThickness' also contribute to the prediction but have a lesser individual impact compared to Glucose, BMI, and Age.

6. Deployment with Streamlit:

The trained LogisticRegression model and the StandardScaler were saved as logistic_regression_model.pkl and scaler.pkl respectively, using joblib.
A Streamlit application (streamlit_app.py) was developed to provide an interactive user interface. This application loads the saved model and scaler, takes user inputs for the 8 features, scales them, and then makes a prediction (Diabetic or Non-Diabetic) along with the probability. Instructions for running the Streamlit app locally were also provided. This deployment allows for an accessible and interactive way for users to test the model with new data.

Overall, this analysis provides a clear understanding of the factors influencing diabetes risk in the dataset, a reasonably performing logistic regression model, and an interactive tool for practical prediction.
