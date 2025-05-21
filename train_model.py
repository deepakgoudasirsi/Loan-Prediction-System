import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def preprocess_data(df):
    # Handle missing values
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(), inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].median(), inplace=True)
    
    # Convert categorical variables to numerical
    le = LabelEncoder()
    categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 
                         'Self_Employed', 'Property_Area', 'Loan_Status']
    
    for column in categorical_columns:
        df[column] = le.fit_transform(df[column])
    
    return df

def train_model():
    # Load the data
    train_df = pd.read_csv('train.csv')
    
    # Preprocess the data
    train_df = preprocess_data(train_df)
    
    # Prepare features and target
    X = train_df.drop(['Loan_ID', 'Loan_Status'], axis=1)
    y = train_df['Loan_Status']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print model performance
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    joblib.dump(model, 'loan_prediction_model.joblib')
    
    # Save the feature names
    joblib.dump(X.columns.tolist(), 'feature_names.joblib')

if __name__ == "__main__":
    train_model()