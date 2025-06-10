import pandas as pd
import numpy as np
import tempfile
import os
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pickle
# strategy I used was to keep asking chatgpt to find faults in data preprocessing where dataset could be mistaken or smth, then coded the fixes
# would be nice if we could have some way to start testing and debugging
def train_model(params):
    dataset_id = params['dataset_csv']
    docs = retrieve_all(dataset_id)
    df = pd.DataFrame(docs)

    input_features = [f.strip() for f in params['inputs'].split(',')]
    target = params['targets'].strip()

    # check columns exist
    missing_cols = [col for col in input_features + [target] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    # separate features and target
    X = df[input_features].copy() #copy so original doesn't get messed w/
    y = df[target].copy()

    # --- 1. Handle missing values ---
    # for numerical columns, fill missing with median
    num_cols = X.select_dtypes(include=[np.number]).columns
    num_imputer = SimpleImputer(strategy='median')
    X[num_cols] = num_imputer.fit_transform(X[num_cols])

    # for categorical columns, fill missing with most frequent
    cat_cols = X.select_dtypes(include=['object', 'category']).columns
    cat_imputer = SimpleImputer(strategy='most_frequent')
    X[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

    # --- 2. Encode categorical variables ---
    # for classification, encode categorical features with LabelEncoder or get_dummies
    # one hot encoding
    X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # --- 3. Encode target if classification and categorical ---
    model_type = params['model']
    if model_type == 'RandomForestClassifier':
        if y.dtype == 'object' or y.dtype.name == 'category':
            le = LabelEncoder()
            y = le.fit_transform(y)
    else:
        # for regression, try to convert target to numeric if not already
        y = pd.to_numeric(y, errors='coerce')
        if y.isnull().any():
            raise ValueError("Target column contains non-numeric values that cannot be converted for regression.")

    # feature scaling???

    # split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train model
    if model_type == 'RandomForestClassifier':
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        metric = 'accuracy'
    elif model_type == 'RandomForestRegressor':
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        metric = 'r2'
    elif model_type == 'XGBRegressor':
        model = LGBMRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        metric = 'r2'
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # just basic routing
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
        pickle.dump(model, tmp)
        tmp_path = tmp.name

    upload_result = insert_temp_file(tmp_path, ext='pkl')
    os.remove(tmp_path)

    return {
        'model_type': model_type,
        'input_features': input_features,
        'target': target,
        'score': score,
        'metric': metric,
        'model_file_url': upload_result['download_url']
    }
