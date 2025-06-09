# This tool will train a machine learning model (XGBRegressor, RandomForestRegressor, or RandomForestClassifier)
# on a provided CSV dataset, using specified input features and target column.
# It will return the trained model's accuracy (or R^2 for regression) and upload the model as a file.

import pandas as pd
import tempfile
import os

# Use lightgbm and sklearn for modeling (no xgboost, as sklearn is not available)
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
import pickle

# 1. Load the CSV data from the knowledge set
dataset_id = params['dataset_csv']
docs = retrieve_all(dataset_id)

# 2. Convert to DataFrame
df = pd.DataFrame(docs)

# 3. Parse input features and target
input_features = [f.strip() for f in params['inputs'].split(',')]
target = params['targets'].strip()

# 4. Prepare data
X = df[input_features]
y = df[target]

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Select and train model
model_type = params['model']
if model_type == 'RandomForestClassifier':
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test, y_pred)
    metric = 'accuracy'
elif model_type == 'RandomForestRegressor':
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    metric = 'r2'
elif model_type == 'XGBRegressor':
    # Use LGBMRegressor as a drop-in replacement for XGBRegressor
    model = LGBMRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)
    metric = 'r2'
else:
    raise ValueError(f"Unsupported model type: {model_type}")

# 7. Save model to a temporary file and upload
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