import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def generate_dataframe(n=1000, random_seed=42):
    np.random.seed(random_seed)
    num_patients = n

    # Baseline: Target Variable with 85/15 split
    cancer_risk = np.random.choice([0, 1], size=num_patients, p=[0.85, 0.15])

    # Independent Features
    # Age: Uniform integers between 25 and 55
    age = np.random.randint(25, 56, size=num_patients)
    # BMI: Floats between 18.0 and 40.0 centered around 26.0 (SD)
    bmi = np.random.normal(loc=26.0, scale=4.0, size=num_patients)
    bmi = np.round(np.clip(bmi, 18.0, 40.0))
    # Family History: 0 or 1 with baseline 10% occurrence
    family_history = np.random.choice([0, 1], size=num_patients, p=[0.90, 0.10])

    df = pd.DataFrame({
        'Age': age,
        'BMI': bmi,
        'Family_History': family_history,
        'Cancer_Risk': cancer_risk
    })

    # Saliva Mammaglobin
    df['Saliva_Mammaglobin'] = np.clip(
        np.where(
            df['Cancer_Risk'] == 0,
            np.random.normal(loc=1.5, scale=0.8, size=num_patients),    # True: Healthy
            np.random.normal(loc=4.2, scale=1.2, size=num_patients)     # False: At-Risk
        ),
        a_min=0.0,
        a_max=None
    )
    df['Saliva_Mammaglobin'] = np.round(df['Saliva_Mammaglobin'], 2)

    print(df.head())
    print('\nTarget Distribution:')
    print(df['Cancer_Risk'].value_counts(normalize=True) * 100) # normalize=True
    print('\nFeature Overlap Verification')
    print(df.groupby('Cancer_Risk')['Saliva_Mammaglobin'].describe().to_string())

    return df


def train_and_eval_model(X, y, random_seed=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)

    # Max depth capped at 5 to prevent data leakage/memorization
    rfc = RandomForestClassifier(n_estimators=100, max_depth=5, class_weight='balanced', random_state=random_seed)
    rfc.fit(X_train, y_train)

    # Evaluation
    y_pred = rfc.predict(X_test)
    y_pred_proba = rfc.predict_proba(X_test)[:, 1]

    print('\nClassification Report')
    print(classification_report(y_test, y_pred, target_names=["Healthy", "At-Risk"]))

    print('\nConfusion Matrix')
    cm_df = pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=['Actual Healthy (0)', 'Actual At-Risk (1)'],
        columns=['Predicted Healthy (0)', 'Predicted At-Risk (1)']
    )
    print(cm_df.to_string())

    return rfc, y_pred_proba


if __name__ == '__main__':
    # DataFrame Generation
    data = generate_dataframe(n=1000)
    X = data.drop('Cancer_Risk', axis=1)
    y = data['Cancer_Risk']

    # Model Training and Evaluation
    rfc_model, y_pred_proba = train_and_eval_model(X, y)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # Dataset Storage
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(data_dir, exist_ok=True)
    filename = 'df_v2.csv'
    filepath = os.path.join(data_dir, filename)

    if os.path.exists(filepath):
        print('\nFILEPATH ALREADY EXISTS.')
        print(f'A file named "{filename}" already exists.')

    else:
        data.to_csv(filepath, index=False)
        print(f'Dataset saved to "{filepath}" successfully.')


    # Model Storage
    model_dir = os.path.join(project_root, 'models')
    os.makedirs(model_dir, exist_ok=True)
    filename = 'rfc_model_v2.pkl'
    filepath = os.path.join(model_dir, filename)

    if os.path.exists(filepath):
        print('\nFILEPATH ALREADY EXISTS.')
        print(f'A file named "{filename}" already exists.')

    else:
        joblib.dump(rfc_model, filepath)
    print(f'Model saved to "{filepath}" successfully.')


    # print("Your data folder is here:", os.path.abspath('./data'))
    # print("Your models folder is here:", os.path.abspath('./models'))
