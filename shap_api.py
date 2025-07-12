import pandas as pd
import shap
from flask import Flask, request, jsonify
from sklearn.ensemble import IsolationForest
import pickle
import os

app = Flask(__name__)

# --- Config ---
MODEL_PATH = 'model.pkl'  # Update as appropriate
DATA_PATH = 'your_data.csv'  # Update as appropriate
ANOMALY_PATH = 'anomalies.csv'  # Update as appropriate

# --- Load Model and Data ---
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return pickle.load(open(MODEL_PATH, 'rb'))

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

def load_anomalies():
    if not os.path.exists(ANOMALY_PATH):
        raise FileNotFoundError(f"Anomaly file not found: {ANOMALY_PATH}")
    return pd.read_csv(ANOMALY_PATH)

model = load_model()


@app.route('/api/anomalies')
def get_anomalies():
    return anomalies.to_json(orient='records')

@app.route('/api/shap/local/<int:anomaly_id>')
def shap_local(anomaly_id):
    row = anomalies.loc[anomalies['log_id'] == anomaly_id]
    if row.empty:
        return jsonify({'error': 'Anomaly not found'}), 404
    shap_val = explainer.shap_values(row)
    return jsonify({
        'shap_values': shap_val.tolist()[0],
        'features': row.iloc[0].to_dict(),
        'feature_names': row.columns.tolist(),
        'expected_value': explainer.expected_value
    })

@app.route('/api/shap/global')
def shap_global():
    global_shap = abs(shap_values).mean(axis=0)
    return jsonify({
        'feature_names': data.columns.tolist(),
        'mean_abs_shap': global_shap.tolist()
    })

@app.route('/api/shap/explanation/<int:anomaly_id>')
def shap_explanation(anomaly_id):
    row = anomalies.loc[anomalies['log_id'] == anomaly_id]
    if row.empty:
        return jsonify({'explanation': 'Anomaly not found.'})
    shap_val = explainer.shap_values(row)
    top_features = sorted(zip(row.columns, shap_val[0]), key=lambda x: abs(x[1]), reverse=True)[:3]
    text = f"Anomaly {anomaly_id} was flagged because: "
    text += ', '.join([f"{f} had a value of {row[f].values[0]} (impact: {s:.2f})" for f, s in top_features])
    return jsonify({'explanation': text})

@app.route('/api/shap/explain', methods=['POST'])
def shap_explain():
    data = request.get_json()
    anomaly = data['anomaly']  # Dict of the anomaly row
    model_type = data.get('model_type', 'Isolation Forest')
    import pickle
    import numpy as np
    import pandas as pd
    import shap
    # Convert to DataFrame for SHAP
    row_df = pd.DataFrame([anomaly])
    features = row_df.select_dtypes(include='number')
    # Load model (Isolation Forest or MLP)
    with open('model.pkl', 'rb') as f:
        model_obj = pickle.load(f)
    if model_type == 'MLP Model' and isinstance(model_obj, tuple):
        model, scaler = model_obj
        features_scaled = scaler.transform(features.values)
        explainer = shap.KernelExplainer(model.predict_proba, scaler.transform(features.values))
        shap_values = explainer.shap_values(features_scaled)
        expected_value = explainer.expected_value[1] if hasattr(explainer, 'expected_value') and isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
        shap_vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    else:
        model = model_obj
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(features)
        expected_value = explainer.expected_value
        shap_vals = shap_values[0] if isinstance(shap_values, np.ndarray) else shap_values[0][0]
    # Build user-friendly explanation (top 3 features)
    friendly_parts = []
    for f, s in top_features:
        value = anomaly.get(f, '?')
        if f == 'amount':
            try:
                value_num = float(value)
            except Exception:
                value_num = None
            if value_num is not None:
                if s < 0:
                    friendly_parts.append(f"This transaction involved a withdrawal of NPR {value_num:,.2f}, which is much higher than usual.")
                else:
                    friendly_parts.append(f"This transaction involved a deposit of NPR {value_num:,.2f}, which is much higher than usual.")
            else:
                friendly_parts.append(f"The transaction amount was unusual.")
        elif f == 'balance':
            friendly_parts.append(f"The account balance ({value}) was unusual compared to typical transactions.")
        elif f == 'transaction_type':
            if str(value).lower() in ['withdrawal', 'withdraw', 'wd']:
                friendly_parts.append("This was a withdrawal transaction, which is less common for this account.")
            elif str(value).lower() in ['deposit', 'dp']:
                friendly_parts.append("This was a deposit transaction, which is less common for this account.")
            else:
                friendly_parts.append(f"The transaction type ('{value}') is rare for this account.")
        elif f == 'device_ip':
            friendly_parts.append(f"The transaction was made from an unusual IP address ({value}).")
        elif f == 'location':
            friendly_parts.append(f"The transaction took place at a location ({value}) that is different from usual.")
        elif f == 'timestamp':
            friendly_parts.append(f"The transaction occurred at an unusual time ({value}).")
        elif f == 'log_id' or f == 'anomaly':
            continue  # skip log_id and anomaly label in friendly explanation
        else:
            friendly_parts.append(f"The value of '{f}' ({value}) was unusual.")
    if friendly_parts:
        friendly_explanation = "Why is this an anomaly? " + " ".join(friendly_parts[:3])
    else:
        friendly_explanation = "This transaction was flagged as an anomaly because it was different from normal transactions in one or more important ways."
    print('FRIENDLY:', friendly_explanation)  # Debug print
    return jsonify({
        'shap_values': shap_vals.tolist(),
        'feature_names': features.columns.tolist(),
        'expected_value': expected_value,
        'explanation': friendly_explanation  # Only user-friendly explanation
    })

if __name__ == '__main__':
    app.run(debug=True)
