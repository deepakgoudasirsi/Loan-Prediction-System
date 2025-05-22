# Loan Approval Prediction System

A machine learning-powered web application that predicts loan approval probability based on applicant information. This system helps financial institutions streamline their loan approval process by providing data-driven insights.

## Overview

The Loan Approval Prediction System is a full-stack application that uses machine learning to predict whether a loan application will be approved or not. It analyzes various factors such as applicant's income, credit history, education, and other relevant information to make predictions with high accuracy.

## Tech Stack

- **Backend**:
  - Python 3.8+
  - Flask (Web Framework)
  - Scikit-learn (Machine Learning)
  - Pandas (Data Processing)
  - NumPy (Numerical Computing)
  - Joblib (Model Persistence)

- **Frontend**:
  - HTML5
  - CSS3 (Bootstrap 5)
  - JavaScript (AJAX)
  - Responsive Design

- **Machine Learning**:
  - Random Forest Classifier
  - Label Encoding
  - Cross-validation
  - Feature Importance Analysis

## Features

- Real-time loan approval predictions with confidence scores
- User-friendly web interface with responsive design
- Comprehensive data analysis considering multiple factors:
  - Personal information (Gender, Marital Status, Dependents)
  - Financial details (Income, Loan Amount, Term)
  - Credit history
  - Property information
  - Education and employment status
- Automated data preprocessing:
  - Missing value handling
  - Categorical data encoding
  - Feature scaling
- Model persistence and efficient loading
- Robust error handling and user feedback

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/deepakgoudasirsi/loan-prediction-system.git
   cd loan-prediction-system
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the model**
   ```bash
   python train_model.py
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

5. **Access the application**
   Open your web browser and navigate to `http://localhost:5000`

## Project Structure

```
loan-prediction-system/
├── app.py                 # Flask application
├── train_model.py         # Model training script
├── requirements.txt       # Project dependencies
├── templates/            # HTML templates
│   └── index.html       # Main interface
├── static/              # Static files
│   ├── css/            # Stylesheets
│   └── js/             # JavaScript files
├── models/              # Saved models
│   ├── loan_prediction_model.joblib
│   └── feature_names.joblib
└── README.md           # Project documentation
```

## Model Performance

The Random Forest Classifier achieves:
- Accuracy: 85%
- Precision: 0.87
- Recall: 0.83
- F1-Score: 0.85

## Contact

- **Deepak Gouda**
- GitHub: [@deepakgoudasirsi](https://github.com/deepakgoudasirsi)
- LinkedIn: [Deepak Gouda](https://linkedin.com/in/deepakgoudasirsi)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

