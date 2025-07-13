from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import os
from collections import Counter
from sklearn.utils import resample
import shap
import numpy as np
import shap_utils  # Add at the top with other imports
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import tempfile
import base64
import io
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from tensorflow import keras
from tensorflow.keras import layers, models
from sklearn.metrics import roc_auc_score
import joblib
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, RepeatVector, TimeDistributed, Dense
from sklearn.neighbors import LocalOutlierFactor
from tensorflow.keras import backend as K
try:
    from prophet import Prophet
    print("Prophet imported successfully")
except ImportError:
    print("Prophet not available - skipping Prophet model")
    Prophet = None

app = Flask(__name__, static_folder='static')
CORS(app)

UPLOAD_FOLDER = 'Uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model_iforest = None
model_mlp = None
X_columns = None
selected_model_type = None
mlp_scaler = None

# --- Multimodel Detection: Autoencoder ---
from flask import Blueprint
multimodel_bp = Blueprint('multimodel', __name__)

import threading
model_locks = {"Autoencoder": threading.Lock(), "LSTM": threading.Lock(), "LOF": threading.Lock(), "Prophet": threading.Lock(), "VAE": threading.Lock(), "SVDD": threading.Lock()}
autoencoder_model = None
autoencoder_threshold = None
autoencoder_features = None
lstm_model = None
lstm_threshold = None
lstm_features = None
lstm_window_size = 10
lof_model = None
lof_features = None
prophet_model = None
prophet_col = 'amount'
prophet_interval_width = 0.95
prophet_anomaly_indices = None
vae_model = None
vae_threshold = None
vae_features = None
svdd_model = None
svdd_center = None
svdd_threshold = None
svdd_features = None

def build_autoencoder(input_dim):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(input_dim // 2, activation='relu'),
        layers.Dense(input_dim // 4, activation='relu'),
        layers.Dense(input_dim // 2, activation='relu'),
        layers.Dense(input_dim, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def create_lstm_sequences(X, window_size):
    seqs = []
    for i in range(len(X) - window_size + 1):
        seqs.append(X[i:i+window_size])
    return np.array(seqs)

def build_lstm_autoencoder(input_dim, window_size):
    model = Sequential([
        LSTM(32, activation='relu', input_shape=(window_size, input_dim), return_sequences=True),
        LSTM(16, activation='relu', return_sequences=False),
        RepeatVector(window_size),
        LSTM(16, activation='relu', return_sequences=True),
        LSTM(32, activation='relu', return_sequences=True),
        TimeDistributed(Dense(input_dim))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_vae(input_dim, latent_dim=2):
    inputs = keras.Input(shape=(input_dim,))
    h = layers.Dense(16, activation='relu')(inputs)
    z_mean = layers.Dense(latent_dim)(h)
    z_log_var = layers.Dense(latent_dim)(h)
    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon
    z = layers.Lambda(sampling)([z_mean, z_log_var])
    encoder = keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(16, activation='relu')(latent_inputs)
    outputs = layers.Dense(input_dim, activation='linear')(x)
    decoder = keras.Model(latent_inputs, outputs, name='decoder')
    outputs = decoder(encoder(inputs)[2])
    vae = keras.Model(inputs, outputs, name='vae')
    reconstruction_loss = keras.losses.mse(inputs, outputs)
    reconstruction_loss = K.sum(reconstruction_loss, axis=1)
    kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)
    vae.add_loss(K.mean(reconstruction_loss + kl_loss))
    vae.compile(optimizer='adam')
    return vae

def build_svdd(input_dim):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(8, activation='relu'),
        layers.Dense(4, activation='relu'),
        layers.Dense(2, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def calculate_risk(score):
    if score > 0.8:
        return 'high'
    elif score > 0.6:
        return 'medium'
    else:
        return 'low'

@multimodel_bp.route('/multimodel/test', methods=['GET'])
def multimodel_test():
    return jsonify({'message': 'Multimodel blueprint is working!'})

@multimodel_bp.route('/multimodel/train', methods=['POST'])
def multimodel_train():
    print("=== MULTIMODEL TRAIN ENDPOINT CALLED ===")
    print("Request data:", request.get_json())
    
    try:
        data = request.get_json()
        filename = data.get('filename')
        if not filename:
            return jsonify({'error': 'Missing filename'}), 400
            
        print(f"Processing file: {filename}")
        
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
            
        df = pd.read_csv(filepath)
        print(f"Loaded CSV with {len(df)} rows")
        
        # Use only numeric columns
        features = df.select_dtypes(include='number').copy()
        if 'label' in features:
            features = features.drop('label', axis=1)
        features = features.fillna(features.mean())
        
        print(f"Training models with {features.shape[1]} features")
        
        # Train simple Isolation Forest
        from sklearn.ensemble import IsolationForest
        iforest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
        iforest.fit(features)
        
        # Train LOF
        from sklearn.neighbors import LocalOutlierFactor
        lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05, novelty=True)
        lof.fit(features)
        
        print("Basic models trained successfully")
        
        # Calculate metrics if labels exist
        metrics = {}
        if 'label' in df.columns:
            y_true = df['label'].values
            iforest_preds = (iforest.predict(features) == -1).astype(int)
            lof_preds = (lof.predict(features) == -1).astype(int)
            
            from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
            
            metrics = {
                'Isolation Forest': {
                    'accuracy': float(accuracy_score(y_true, iforest_preds)),
                    'precision': float(precision_score(y_true, iforest_preds, zero_division=0)),
                    'recall': float(recall_score(y_true, iforest_preds, zero_division=0)),
                    'f1': float(f1_score(y_true, iforest_preds, zero_division=0))
                },
                'LOF': {
                    'accuracy': float(accuracy_score(y_true, lof_preds)),
                    'precision': float(precision_score(y_true, lof_preds, zero_division=0)),
                    'recall': float(recall_score(y_true, lof_preds, zero_division=0)),
                    'f1': float(f1_score(y_true, lof_preds, zero_division=0))
                }
            }
        else:
            metrics = {
                'Isolation Forest': {'status': 'trained'},
                'LOF': {'status': 'trained'}
            }
        
        # Save models
        import joblib
        joblib.dump(iforest, 'iforest_model.pkl')
        joblib.dump(lof, 'lof_model.pkl')
        
        return jsonify({
            'message': 'Basic models trained successfully',
            'model_types': ['Isolation Forest', 'LOF'],
            'metrics': metrics,
            'features': features.columns.tolist()
        })
        
    except Exception as e:
        print(f"Error in multimodel_train: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Training failed: {str(e)}'}), 500

@multimodel_bp.route('/multimodel/predict', methods=['POST'])
def multimodel_predict():
    print("=== MULTIMODEL PREDICT ENDPOINT CALLED ===")
    print("Request data:", request.get_json())
    
    try:
        data = request.get_json()
        filename = data.get('filename')
        if not filename:
            return jsonify({'error': 'Missing filename'}), 400
            
        print(f"Processing file: {filename}")
        
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
            
        df = pd.read_csv(filepath)
        print(f"Loaded CSV with {len(df)} rows")
        
        # Use only numeric columns
        features = df.select_dtypes(include='number').copy()
        if 'label' in features:
            features = features.drop('label', axis=1)
        features = features.fillna(features.mean())
        X = features.values
        
        print(f"Features shape: {X.shape}")
        
        # Initialize predictions arrays and scores
        iforest_pred = np.zeros(len(df))
        lof_pred = np.zeros(len(df))
        iforest_score = np.zeros(len(df))
        lof_score = np.zeros(len(df))
        autoencoder_score = np.zeros(len(df))
        lstm_score = np.zeros(len(df))
        vae_score = np.zeros(len(df))
        svdd_score = np.zeros(len(df))
        model_scores_dict = {}
        
        # --- Isolation Forest ---
        try:
            if os.path.exists('iforest_model.pkl'):
                print("Loading Isolation Forest model...")
                iforest = joblib.load('iforest_model.pkl')
                iforest_pred = (iforest.predict(X) == -1).astype(int)
                iforest_score_raw = -iforest.decision_function(X)
                iforest_score = (iforest_score_raw - iforest_score_raw.min()) / (iforest_score_raw.max() - iforest_score_raw.min() + 1e-8)
                model_scores_dict['Isolation Forest'] = iforest_score
                print(f"Isolation Forest detected {np.sum(iforest_pred)} anomalies")
            else:
                print("Isolation Forest model not found, skipping...")
        except Exception as e:
            print(f"Error with Isolation Forest: {e}")
        
        # --- LOF ---
        try:
            if os.path.exists('lof_model.pkl'):
                print("Loading LOF model...")
                lof_model = joblib.load('lof_model.pkl')
                lof_pred = (lof_model.predict(X) == -1).astype(int)
                lof_score_raw = -lof_model.decision_function(X)
                lof_score = (lof_score_raw - lof_score_raw.min()) / (lof_score_raw.max() - lof_score_raw.min() + 1e-8)
                model_scores_dict['LOF'] = lof_score
                print(f"LOF detected {np.sum(lof_pred)} anomalies")
            else:
                print("LOF model not found, skipping...")
        except Exception as e:
            print(f"Error with LOF: {e}")
        
        # --- Autoencoder ---
        try:
            if os.path.exists('autoencoder_model.h5') and os.path.exists('autoencoder_meta.pkl'):
                print("Loading Autoencoder model...")
                from tensorflow.keras.models import load_model
                autoencoder = load_model('autoencoder_model.h5')
                meta = joblib.load('autoencoder_meta.pkl')
                autoencoder_features = meta['features']
                X_auto = features[autoencoder_features].values
                recon = autoencoder.predict(X_auto)
                mse = ((X_auto - recon) ** 2).mean(axis=1)
                autoencoder_score = (mse - mse.min()) / (mse.max() - mse.min() + 1e-8)
                model_scores_dict['Autoencoder'] = autoencoder_score
                print("Autoencoder scores computed.")
            else:
                print("Autoencoder model or meta not found, skipping...")
        except Exception as e:
            print(f"Error with Autoencoder: {e}")
        
        # --- LSTM Autoencoder ---
        try:
            if os.path.exists('lstm_autoencoder_model.h5') and os.path.exists('lstm_autoencoder_meta.pkl') and 'timestamp' in df.columns:
                print("Loading LSTM Autoencoder model...")
                from tensorflow.keras.models import load_model
                lstm_model = load_model('lstm_autoencoder_model.h5')
                meta = joblib.load('lstm_autoencoder_meta.pkl')
                lstm_features = meta['features']
                window_size = meta.get('window_size', 10)
                X_lstm = df.sort_values('timestamp')[lstm_features].fillna(features.mean()).values
                def create_lstm_sequences(X, window_size):
                    seqs = []
                    for i in range(len(X) - window_size + 1):
                        seqs.append(X[i:i+window_size])
                    return np.array(seqs)
                seqs = create_lstm_sequences(X_lstm, window_size)
                recon = lstm_model.predict(seqs)
                mse_seq = ((seqs - recon) ** 2).mean(axis=(1,2))
                # Map sequence-level scores to row-level (repeat for each window)
                lstm_score_seq = (mse_seq - mse_seq.min()) / (mse_seq.max() - mse_seq.min() + 1e-8)
                lstm_score = np.zeros(len(df))
                for i in range(len(lstm_score_seq)):
                    lstm_score[i:i+window_size] = np.maximum(lstm_score[i:i+window_size], lstm_score_seq[i])
                model_scores_dict['LSTM Autoencoder'] = lstm_score
                print("LSTM Autoencoder scores computed.")
            else:
                print("LSTM Autoencoder model or meta not found, skipping...")
        except Exception as e:
            print(f"Error with LSTM Autoencoder: {e}")
        
        # --- VAE ---
        try:
            if os.path.exists('vae_model.h5') and os.path.exists('vae_meta.pkl'):
                print("Loading VAE model...")
                from tensorflow.keras.models import load_model
                vae = load_model('vae_model.h5', compile=False)
                meta = joblib.load('vae_meta.pkl')
                vae_features = meta['features']
                X_vae = features[vae_features].values
                recon = vae.predict(X_vae)
                mse = ((X_vae - recon) ** 2).mean(axis=1)
                vae_score = (mse - mse.min()) / (mse.max() - mse.min() + 1e-8)
                model_scores_dict['VAE'] = vae_score
                print("VAE scores computed.")
            else:
                print("VAE model or meta not found, skipping...")
        except Exception as e:
            print(f"Error with VAE: {e}")
        
        # --- Deep SVDD ---
        try:
            if os.path.exists('svdd_model.h5') and os.path.exists('svdd_meta.pkl'):
                print("Loading Deep SVDD model...")
                from tensorflow.keras.models import load_model
                svdd = load_model('svdd_model.h5', compile=False)
                meta = joblib.load('svdd_meta.pkl')
                svdd_features = meta['features']
                svdd_center = meta['center']
                svdd_threshold = meta['threshold']
                X_svdd = features[svdd_features].values
                outputs = svdd.predict(X_svdd)
                dists = np.linalg.norm(outputs - svdd_center, axis=1)
                svdd_score = (dists - dists.min()) / (dists.max() - dists.min() + 1e-8)
                model_scores_dict['Deep SVDD'] = svdd_score
                print("Deep SVDD scores computed.")
            else:
                print("Deep SVDD model or meta not found, skipping...")
        except Exception as e:
            print(f"Error with Deep SVDD: {e}")
        
        # --- Ensemble: flag as anomaly if any model detects it ---
        combined_pred = (iforest_pred | lof_pred)
        print(f"Ensemble detected {np.sum(combined_pred)} anomalies")
        
        # If labels exist, compute metrics
        metrics = {}
        if 'label' in df:
            from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
            y_true = df['label'].values
            metrics = {
                'ensemble': {
                    'accuracy': float(accuracy_score(y_true, combined_pred)),
                    'precision': float(precision_score(y_true, combined_pred, zero_division=0)),
                    'recall': float(recall_score(y_true, combined_pred, zero_division=0)),
                    'f1': float(f1_score(y_true, combined_pred, zero_division=0))
                },
                'Isolation Forest': {
                    'accuracy': float(accuracy_score(y_true, iforest_pred)),
                    'precision': float(precision_score(y_true, iforest_pred, zero_division=0)),
                    'recall': float(recall_score(y_true, iforest_pred, zero_division=0)),
                    'f1': float(f1_score(y_true, iforest_pred, zero_division=0))
                },
                'LOF': {
                    'accuracy': float(accuracy_score(y_true, lof_pred)),
                    'precision': float(precision_score(y_true, lof_pred, zero_division=0)),
                    'recall': float(recall_score(y_true, lof_pred, zero_division=0)),
                    'f1': float(f1_score(y_true, lof_pred, zero_division=0))
                }
            }
        
        # Build anomalies output
        anomalies = []
        for i, row in df.iterrows():
            if combined_pred[i] == 1:
                anomaly = row.to_dict()
                anomaly['iforest_flag'] = int(iforest_pred[i])
                anomaly['lof_flag'] = int(lof_pred[i])
                anomaly['ensemble_flag'] = int(combined_pred[i])
                # Collect available model scores for this row
                per_model_scores = {k: float(v[i]) for k, v in model_scores_dict.items() if len(v) == len(df)}
                anomaly['per_model_scores'] = per_model_scores
                # Ensemble score: mean of all available model scores
                if per_model_scores:
                    ensemble_score = np.mean(list(per_model_scores.values()))
                    anomaly['anomaly_score'] = float(ensemble_score)
                    # Use Isolation Forest score for risk calculation to match single-model section
                    iforest_score = per_model_scores.get('Isolation Forest', ensemble_score)
                    anomaly['risk'] = calculate_risk(iforest_score)
                else:
                    anomaly['anomaly_score'] = 0.0
                    anomaly['risk'] = 'low'
                anomaly['models'] = []
                if iforest_pred[i]: anomaly['models'].append('Isolation Forest')
                if lof_pred[i]: anomaly['models'].append('LOF')
                anomalies.append(anomaly)
        print(f"Returning {len(anomalies)} anomalies")
        return jsonify({
            'anomalies': anomalies,
            'metrics': metrics,
            'model_types': list(model_scores_dict.keys()),
            'features': list(features.columns)
        })
    except Exception as e:
        print(f"Error in multimodel_predict: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

app.register_blueprint(multimodel_bp)

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/shap.html')
def shap_explainer():
    return render_template('shap.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory('templates', path)

@app.route('/upload', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    print(f"Saving file to: {os.path.abspath(filepath)}")  # Debug: log the save path
    file.save(filepath)
    return jsonify({'message': 'File uploaded successfully', 'filename': file.filename})

@app.route('/train', methods=['POST'])
def train_model():
    data = request.get_json()
    filename = data.get('filename')
    model_type = data.get('model_type', 'Isolation Forest')
    tune = data.get('tune', False)  # New: allow tuning via request
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    df = pd.read_csv(filepath)
    # Use all numeric columns except 'label' for training
    features = df.select_dtypes(include='number').copy()
    if 'label' in features:
        features = features.drop('label', axis=1)
    # Fill missing values with column mean
    features = features.fillna(features.mean())
    global model_iforest, model_mlp, X_columns, selected_model_type
    X_columns = features.columns.tolist()
    metrics = {}
    best_params = None
    if model_type == 'Isolation Forest':
        if tune:
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_samples': ['auto', 0.8, 0.9],
                'contamination': [0.01, 0.05, 0.1],
                'max_features': [1.0, 0.8, 0.6]
            }
            grid = GridSearchCV(IsolationForest(random_state=42), param_grid, scoring='f1', cv=3, n_jobs=-1)
            if 'label' in df:
                y_true = df['label'].values
                grid.fit(features, y_true)
            else:
                grid.fit(features)
            model_iforest = grid.best_estimator_
            best_params = grid.best_params_
        else:
            model_iforest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
            model_iforest.fit(features)
        selected_model_type = 'Isolation Forest'
        # If labels are present, calculate metrics
        if 'label' in df:
            preds = model_iforest.predict(features)
            preds = (preds == -1).astype(int)
            y_true = df['label'].values
            metrics = {
                'accuracy': accuracy_score(y_true, preds),
                'precision': precision_score(y_true, preds, zero_division=0),
                'recall': recall_score(y_true, preds, zero_division=0),
                'f1': f1_score(y_true, preds, zero_division=0)
            }
    elif model_type == 'MLP Model':
        if 'label' not in df:
            return jsonify({'error': 'No label column for supervised training'}), 400
        scaler = StandardScaler()
        X = features.values
        y = df['label'].values
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        class_dist = dict(Counter(y_train))
        X_train_df = pd.DataFrame(X_train)
        y_train_df = pd.Series(y_train)
        Xy_train = X_train_df.copy()
        Xy_train['label'] = y_train_df.values
        majority = Xy_train[Xy_train['label'] == 0]
        minority = Xy_train[Xy_train['label'] == 1]
        if len(minority) > 0 and len(majority) > 0:
            minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
            upsampled = pd.concat([majority, minority_upsampled])
            X_train_bal = upsampled.drop('label', axis=1).values
            y_train_bal = upsampled['label'].values
        else:
            X_train_bal = X_train
            y_train_bal = y_train
        if tune:
            param_grid = {
                'hidden_layer_sizes': [(64, 32), (128, 64), (128, 64, 32)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate_init': [0.001, 0.01],
                'max_iter': [300, 500, 800]
            }
            grid = GridSearchCV(MLPClassifier(random_state=42, early_stopping=True, n_iter_no_change=10), param_grid, scoring='f1', cv=3, n_jobs=-1)
            grid.fit(X_train_bal, y_train_bal)
            model_mlp = grid.best_estimator_
            best_params = grid.best_params_
        else:
            model_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, alpha=0.001, random_state=42, early_stopping=True, n_iter_no_change=10)
            model_mlp.fit(X_train_bal, y_train_bal)
        selected_model_type = 'MLP Model'
        preds = model_mlp.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        recall_val = recall_score(y_test, preds, zero_division=0)
        metrics = {
            'accuracy': accuracy_score(y_test, preds),
            'precision': precision_score(y_test, preds, zero_division=0),
            'recall': recall_val,
            'f1': f1_score(y_test, preds, zero_division=0),
            'confusion_matrix': cm.tolist(),
            'class_distribution': {str(k): int(v) for k, v in class_dist.items()},
            'train_class_distribution': {str(k): int(v) for k, v in Counter(y_train_bal).items()}
        }
        if recall_val == 0:
            metrics['warning'] = 'Model did not detect any anomalies in the test set. Consider more feature engineering.'
        global mlp_scaler
        mlp_scaler = scaler
    else:
        return jsonify({'error': 'Unknown model type'}), 400
    response = {'message': 'Model trained', 'features': X_columns, 'model_type': selected_model_type, 'metrics': metrics}
    if best_params is not None:
        response['best_params'] = best_params
    return jsonify(response)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    filename = data.get('filename')
    model_type = data.get('model_type', 'Isolation Forest')

    # Load data
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    df = pd.read_csv(filepath)

    # Use more features for better accuracy
    features = ['amount', 'balance', 'transaction_type', 'branch_name']
    X = df[features].copy()

    # Encode categorical features
    categorical = ['transaction_type', 'branch_name']
    numeric = ['amount', 'balance']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical)
        ]
    )

    # Build pipeline
    pipeline = Pipeline([
        ('pre', preprocessor),
        ('clf', IsolationForest(contamination=0.05, n_estimators=200, random_state=42))
    ])

    pipeline.fit(X)

    # Get anomaly scores and predictions
    X_trans = pipeline.named_steps['pre'].transform(X)
    scores = -pipeline.named_steps['clf'].decision_function(X_trans)
    preds = pipeline.named_steps['clf'].predict(X_trans)

    # Normalize scores
    norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    # If you have labels, evaluate
    metrics = {}
    if 'label' in df.columns:
        y_true = df['label'].values
        y_pred = (preds == -1).astype(int)
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred)),
            'precision': float(precision_score(y_true, y_pred, zero_division=0)),
            'recall': float(recall_score(y_true, y_pred, zero_division=0)),
            'f1': float(f1_score(y_true, y_pred, zero_division=0))
        }

    # Build output
    anomalies = []
    for i, row in df.iterrows():
        if preds[i] == -1:  # Only anomalies
            anomaly = row.to_dict()
            anomaly['anomaly_score'] = float(norm_scores[i])
            anomaly['risk'] = calculate_risk(anomaly['anomaly_score'])
            anomalies.append(anomaly)

    # Prepare chart data for frontend
    all_amounts = df['amount'].tolist() if 'amount' in df else []
    anomaly_indices = [i for i, p in enumerate(preds) if p == -1]
    anomaly_amounts = [df.iloc[i]['amount'] for i in anomaly_indices]

    chart_data = {
        'amounts': all_amounts,
        'anomaly_indices': anomaly_indices,
        'anomaly_amounts': anomaly_amounts
    }

    return jsonify({
        'anomalies': anomalies,
        'metrics': metrics,
        'model_type': model_type,
        'features': features,
        'chart_data': chart_data
    })

@app.route('/data', methods=['GET'])
def get_csv_data():
    filename = request.args.get('filename')
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    try:
        df = pd.read_csv(filepath)
        data = df.to_dict(orient='records')
        return jsonify({'data': data})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/shap', methods=['POST'])
def shap_explain():
    try:
        data = request.get_json()
        filename = data.get('filename')
        model_type = data.get('model_type', 'Isolation Forest')
        index = data.get('index', 0)

        # Check file exists
        file_path = os.path.join('Uploads', filename) if filename else None
        if not filename or not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 400

        # Load data
        df = pd.read_csv(file_path)
        features = df.select_dtypes(include='number').copy()
        if 'label' in features:
            features = features.drop('label', axis=1)
        features = features.fillna(features.mean())
        feature_names = list(features.columns)
        feature_values = features.iloc[index].to_dict() if index < len(features) else {}

        # Load model
        global model_iforest, model_mlp, mlp_scaler
        if model_type == 'Isolation Forest' and model_iforest is not None:
            import shap_utils
            local_shap = shap_utils.compute_tree_shap_local(model_iforest, features, index).tolist()
            base_value = 0  # For Isolation Forest, base value is usually 0
            # User-friendly explanation
            main_feature = feature_names[int(np.argmax(np.abs(local_shap)))]
            if main_feature == 'amount':
                explanation = "This transaction was flagged as an anomaly because the withdrawal or deposit amount was much higher than usual."
            elif main_feature == 'device_ip':
                explanation = "This transaction was flagged as an anomaly because it was made from an unusual IP address."
            elif main_feature == 'location':
                explanation = "This transaction was flagged as an anomaly because it took place at a location that is different from usual."
            elif main_feature == 'timestamp':
                explanation = "This transaction was flagged as an anomaly because it occurred at an unusual time."
            else:
                explanation = f"This transaction was flagged as an anomaly because the value of '{main_feature}' was unusual."
        elif model_type == 'MLP Model' and model_mlp is not None:
            import shap_utils
            X_scaled = mlp_scaler.transform(features.values)
            local_shap = shap_utils.compute_mlp_shap_local(model_mlp, X_scaled, index).tolist()
            base_value = 0  # For MLP, base value is usually 0 for anomaly score
            main_feature = feature_names[int(np.argmax(np.abs(local_shap)))]
            if main_feature == 'amount':
                explanation = "This transaction was flagged as an anomaly because the withdrawal or deposit amount was much higher than usual."
            elif main_feature == 'device_ip':
                explanation = "This transaction was flagged as an anomaly because it was made from an unusual IP address."
            elif main_feature == 'location':
                explanation = "This transaction was flagged as an anomaly because it took place at a location that is different from usual."
            elif main_feature == 'timestamp':
                explanation = "This transaction was flagged as an anomaly because it occurred at an unusual time."
            else:
                explanation = f"This transaction was flagged as an anomaly because the value of '{main_feature}' was unusual."
        else:
            return jsonify({'error': 'Model not trained or unknown model type'}), 400

        # Waterfall plot (simple version)
        y = [base_value]
        for val in local_shap:
            y.append(y[-1] + val)
        import plotly.graph_objs as go
        waterfall_data = [go.Scatter(
            x=['Base'] + feature_names,
            y=y,
            mode='lines+markers',
            marker=dict(color='rgba(239, 68, 68, 0.8)')
        )]
        waterfall_layout = go.Layout(
            title='SHAP Waterfall (Cumulative)',
            margin=dict(l=60)
        )

        return jsonify({
            'feature_names': feature_names,
            'shap_values': local_shap,
            'waterfall_plot': {
                'data': [trace.to_plotly_json() for trace in waterfall_data],
                'layout': waterfall_layout.to_plotly_json()
            },
            'explanation': explanation
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/shap_global', methods=['POST'])
def shap_global_explain():
    try:
        data = request.get_json()
        filename = data.get('filename')
        model_type = data.get('model_type', 'Isolation Forest')
        file_path = os.path.join('Uploads', filename) if filename else None
        if not filename or not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 400
        df = pd.read_csv(file_path)
        features = df.select_dtypes(include='number').copy()
        if 'label' in features:
            features = features.drop('label', axis=1)
        features = features.fillna(features.mean())
        feature_names = list(features.columns)
        global model_iforest, model_mlp, mlp_scaler
        if model_type == 'Isolation Forest' and model_iforest is not None:
            import shap_utils
            explainer = shap.TreeExplainer(model_iforest)
            shap_values = explainer.shap_values(features)
            global_importance = np.abs(shap_values).mean(axis=0)
        elif model_type == 'MLP Model' and model_mlp is not None:
            import shap_utils
            X_scaled = mlp_scaler.transform(features.values)
            global_importance = shap_utils.compute_mlp_shap_global(model_mlp, X_scaled).tolist()
        else:
            return jsonify({'error': 'Model not trained or unknown model type'}), 400
        return jsonify({
            'feature_names': feature_names,
            'global_importance': global_importance
        })
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/shap_plot', methods=['POST'])
def shap_plot():
    """
    Expects JSON: {
        'filename': str,
        'model_type': str,
        'plot_type': 'summary' | 'force' | 'dependence',
        'index': int (for force plot),
        'feature': str or int (for dependence plot),
        'summary_plot_type': 'bar' | 'dot' (optional, for summary)
    }
    Returns: { 'image_base64': str }
    """
    import shap_utils
    data = request.get_json()
    filename = data.get('filename')
    model_type = data.get('model_type', 'Isolation Forest')
    plot_type = data.get('plot_type', 'summary')
    index = data.get('index', None)
    feature = data.get('feature', None)
    summary_plot_type = data.get('summary_plot_type', 'bar')
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    df = pd.read_csv(filepath)
    features = df.select_dtypes(include='number').copy()
    if 'label' in features:
        features = features.drop('label', axis=1)
    features = features.fillna(features.mean())
    global model_iforest, model_mlp, mlp_scaler
    try:
        if model_type == 'Isolation Forest' and model_iforest is not None:
            global_importance, shap_values = shap_utils.compute_tree_shap_global(model_iforest, features)
            feature_names = list(features.columns)
            if plot_type == 'summary':
                img_b64 = shap_utils.shap_summary_plot(shap_values, features, feature_names, plot_type=summary_plot_type)
            elif plot_type == 'force':
                explainer = shap.TreeExplainer(model_iforest)
                if index is None:
                    return jsonify({'error': 'Index required for force plot'}), 400
                img_b64 = shap_utils.shap_force_plot(explainer, shap_values, features, feature_names, index)
            elif plot_type == 'dependence':
                if feature is None:
                    return jsonify({'error': 'Feature required for dependence plot'}), 400
                img_b64 = shap_utils.shap_dependence_plot(shap_values, features, feature_names, feature)
            else:
                return jsonify({'error': 'Unknown plot type'}), 400
        elif model_type == 'MLP Model' and model_mlp is not None:
            X_scaled = mlp_scaler.transform(features.values)
            global_importance, shap_values, sample_idx = shap_utils.compute_mlp_shap_global(model_mlp, X_scaled)
            feature_names = list(features.columns)
            if plot_type == 'summary':
                img_b64 = shap_utils.shap_summary_plot(shap_values[1], X_scaled[sample_idx], feature_names, plot_type=summary_plot_type)
            elif plot_type == 'force':
                background = shap.utils.sample(X_scaled, min(100, X_scaled.shape[0]), random_state=42)
                explainer = shap.KernelExplainer(model_mlp.predict_proba, background)
                if index is None:
                    return jsonify({'error': 'Index required for force plot'}), 400
                img_b64 = shap_utils.shap_force_plot(explainer, shap_values[1], X_scaled[sample_idx], feature_names, index)
            elif plot_type == 'dependence':
                if feature is None:
                    return jsonify({'error': 'Feature required for dependence plot'}), 400
                img_b64 = shap_utils.shap_dependence_plot(shap_values[1], X_scaled[sample_idx], feature_names, feature)
            else:
                return jsonify({'error': 'Unknown plot type'}), 400
        else:
            return jsonify({'error': 'Model not trained or unknown model type'}), 400
        return jsonify({'image_base64': img_b64})
    except Exception as e:
        return jsonify({'error': f'SHAP plot generation failed: {str(e)}'}), 500

@app.route('/shap_report', methods=['POST'])
def shap_report():
    """
    Expects JSON: {
        'filename': str,
        'model_type': str,
        'indices': list of int (optional, for local explanations)
    }
    Returns: PDF file (application/pdf)
    """
    import shap_utils
    data = request.get_json()
    filename = data.get('filename')
    model_type = data.get('model_type', 'Isolation Forest')
    indices = data.get('indices', [])
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    df = pd.read_csv(filepath)
    features = df.select_dtypes(include='number').copy()
    if 'label' in features:
        features = features.drop('label', axis=1)
    features = features.fillna(features.mean())
    global model_iforest, model_mlp, mlp_scaler
    try:
        # Prepare SHAP values and plots
        if model_type == 'Isolation Forest' and model_iforest is not None:
            global_importance, shap_values = shap_utils.compute_tree_shap_global(model_iforest, features)
            feature_names = list(features.columns)
            summary_b64 = shap_utils.shap_summary_plot(shap_values, features, feature_names, plot_type='bar')
            explainer = shap.TreeExplainer(model_iforest)
            local_b64s = []
            for idx in indices:
                local_b64s.append(shap_utils.shap_force_plot(explainer, shap_values, features, feature_names, idx))
        elif model_type == 'MLP Model' and model_mlp is not None:
            X_scaled = mlp_scaler.transform(features.values)
            global_importance, shap_values, sample_idx = shap_utils.compute_mlp_shap_global(model_mlp, X_scaled)
            feature_names = list(features.columns)
            summary_b64 = shap_utils.shap_summary_plot(shap_values[1], X_scaled[sample_idx], feature_names, plot_type='bar')
            background = shap.utils.sample(X_scaled, min(100, X_scaled.shape[0]), random_state=42)
            explainer = shap.KernelExplainer(model_mlp.predict_proba, background)
            local_b64s = []
            for idx in indices:
                local_b64s.append(shap_utils.shap_force_plot(explainer, shap_values[1], X_scaled[sample_idx], feature_names, idx))
        else:
            return jsonify({'error': 'Model not trained or unknown model type'}), 400
        # Generate PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmpfile:
            c = canvas.Canvas(tmpfile.name, pagesize=letter)
            width, height = letter
            c.setFont('Helvetica-Bold', 16)
            c.drawString(40, height-40, 'SHAP Explainability Report')
            c.setFont('Helvetica', 12)
            c.drawString(40, height-70, f'Model: {model_type}')
            c.drawString(40, height-90, f'File: {filename}')
            c.drawString(40, height-110, 'Global SHAP Summary:')
            # Add summary plot
            summary_img = ImageReader(io.BytesIO(base64.b64decode(summary_b64)))
            c.drawImage(summary_img, 40, height-400, width=500, height=250, preserveAspectRatio=True, mask='auto')
            y = height-420
            if local_b64s:
                c.drawString(40, y-20, 'Local SHAP Explanations:')
                for i, local_b64 in enumerate(local_b64s):
                    y -= 270
                    if y < 100:
                        c.showPage()
                        y = height-100
                    c.drawString(40, y, f'Instance {indices[i]}:')
                    local_img = ImageReader(io.BytesIO(base64.b64decode(local_b64)))
                    c.drawImage(local_img, 40, y-220, width=500, height=200, preserveAspectRatio=True, mask='auto')
            c.save()
            tmpfile.seek(0)
            pdf_bytes = tmpfile.read()
        return (pdf_bytes, 200, {'Content-Type': 'application/pdf', 'Content-Disposition': f'attachment; filename=shap_report_{model_type.replace(" ", "_")}.pdf'})
    except Exception as e:
        return jsonify({'error': f'SHAP report generation failed: {str(e)}'}), 500

@app.route('/api/shap/explain', methods=['POST'])
def api_shap_explain():
    import traceback
    import logging
    try:
        data = request.get_json(force=True, silent=True)
        app.logger.info(f"/api/shap/explain received: {data}")
        if not data:
            app.logger.error("No JSON data received in request.")
            return jsonify({'error': 'No JSON data received in request.'}), 400
        anomaly = data.get('anomaly')
        model_type = data.get('model_type', 'Isolation Forest')
        if anomaly is None or not isinstance(anomaly, dict):
            app.logger.error(f"Missing or invalid 'anomaly' in request: {anomaly}")
            return jsonify({'error': "Missing or invalid 'anomaly' in request."}), 400
        if not model_type:
            app.logger.error("Missing 'model_type' in request.")
            return jsonify({'error': "Missing 'model_type' in request."}), 400

        # Use the anomaly dict directly for SHAP explanation
        import shap
        import numpy as np
        import pandas as pd
        feature_names = list(anomaly.keys())
        feature_values = [anomaly[f] for f in feature_names if isinstance(anomaly[f], (int, float))]
        if not feature_values:
            app.logger.error("No numeric features in anomaly.")
            return jsonify({'error': 'No numeric features in anomaly'}), 400

        global model_iforest, model_mlp, mlp_scaler
        if model_type == 'Isolation Forest' and model_iforest is not None:
            explainer = shap.TreeExplainer(model_iforest)
            row_df = pd.DataFrame([anomaly])
            features = row_df.select_dtypes(include='number')
            feature_values = row_df.iloc[0].to_dict()  # Ensure this is a dict
            shap_vals = explainer.shap_values(features)[0]
            expected_value = explainer.expected_value
        elif model_type == 'MLP Model' and model_mlp is not None:
            row_df = pd.DataFrame([anomaly])
            features = row_df.select_dtypes(include='number')
            X_scaled = mlp_scaler.transform(features.values)
            explainer = shap.KernelExplainer(model_mlp.predict_proba, mlp_scaler.transform(features.values))
            shap_values = explainer.shap_values(X_scaled)
            expected_value = explainer.expected_value[1] if hasattr(explainer, 'expected_value') and isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            shap_vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        else:
            app.logger.error(f"Model not trained or unknown model type: {model_type}")
            return jsonify({'error': 'Model not trained or unknown model type'}), 400

        top_features = sorted(zip(features.columns, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:3]
        # Improved user-friendly AI explanation logic
        # List of fields to ignore in explanations
        ignore_fields = {'log_id', 'anomaly', 'label', 'id', 'index', 'member_id'}
        reasons = []
        # Detect transaction type
        transaction_type = anomaly.get('transaction_type', '').lower()
        amount = None
        try:
            amount = float(anomaly.get('amount', 0))
        except Exception:
            amount = None
        for f, s in top_features:
            if f in ignore_fields:
                continue  # Skip technical fields
            value = feature_values.get(f, '?')
            # Amount logic
            if f == 'amount' and amount is not None:
                if transaction_type:
                    if transaction_type == 'deposit' and abs(s) > 0.1:
                        reasons.append(
                            f"the deposit amount (NPR {amount:,.2f}) was much higher than your usual deposits. "
                            "Such a large deposit is rare for your account and may indicate unusual or unexpected activity."
                        )
                    elif transaction_type == 'withdrawal' and abs(s) > 0.1:
                        reasons.append(
                            f"the withdrawal amount (NPR {abs(amount):,.2f}) was much higher than your usual withdrawals. "
                            "Such a large withdrawal is rare for your account and may indicate unusual or unexpected activity."
                        )
                else:
                    # Fallback: infer from sign
                    if amount > 0 and abs(s) > 0.1:
                        reasons.append(
                            f"the deposit amount (NPR {amount:,.2f}) was much higher than your usual deposits. "
                            "Such a large deposit is rare for your account and may indicate unusual or unexpected activity."
                        )
                    elif amount < 0 and abs(s) > 0.1:
                        reasons.append(
                            f"the withdrawal amount (NPR {abs(amount):,.2f}) was much higher than your usual withdrawals. "
                            "Such a large withdrawal is rare for your account and may indicate unusual or unexpected activity."
                        )
            # Location
            elif f == 'location' and value and abs(s) > 0.1:
                reasons.append(
                    f"the transaction location ({value}) was different from your usual locations. "
                    "Transactions from new or unexpected locations may indicate unusual activity."
                )
            # Device/IP
            elif f in ['device_ip', 'ip', 'ip_address'] and value and abs(s) > 0.1:
                reasons.append(
                    f"the transaction was made from an unusual IP address ({value}). "
                    "Access from a new device or network may indicate suspicious activity."
                )
            # Time
            elif f in ['timestamp', 'time', 'date'] and value and abs(s) > 0.1:
                reasons.append(
                    f"the transaction occurred at an unusual time ({value}) compared to your normal activity. "
                    "Transactions at odd hours may indicate unexpected or risky behavior."
                )
            # Transaction type
            elif f == 'transaction_type' and value and abs(s) > 0.1:
                reasons.append(
                    f"the transaction type ('{value}') is rare for this account. "
                    "Unusual transaction types may indicate unexpected activity."
                )
            # Fallback for other features
            elif abs(s) > 0.1:
                reasons.append(f"the value of '{f}' ({value}) was unusual compared to your normal transactions.")
        if reasons:
            explanation = "This transaction was flagged as an anomaly because " + ", and ".join(reasons[:3]) + "."
        else:
            explanation = "This transaction was flagged as an anomaly because it was different from normal transactions in one or more important ways."
        # Ensure all returned values are JSON serializable
        shap_vals_list = shap_vals.tolist() if hasattr(shap_vals, 'tolist') else list(shap_vals)
        feature_names_list = list(feature_names) if hasattr(feature_names, '__iter__') else [feature_names]
        expected_value_serializable = float(expected_value) if hasattr(expected_value, '__float__') else expected_value
        return jsonify({
            'shap_values': shap_vals_list,
            'feature_names': feature_names_list,
            'expected_value': expected_value_serializable,
            'explanation': explanation
        })
    except Exception as e:
        tb = traceback.format_exc()
        app.logger.error(f"Exception in /api/shap/explain: {tb}")
        return jsonify({'error': str(e), 'traceback': tb}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)