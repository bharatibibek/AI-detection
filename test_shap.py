#!/usr/bin/env python3
"""
Test script for SHAP functionality
"""

import requests
import json
import os
import pandas as pd
from sklearn.ensemble import IsolationForest
import pickle

def test_shap_functionality():
    """Test the SHAP endpoints"""
    
    # Base URL
    base_url = "http://127.0.0.1:5000"
    
    # Test file
    test_file = "kathmandu_cooperative_logs_20250702_101219.csv"
    
    print("Testing SHAP functionality...")
    
    # 1. First, train a model
    print("1. Training model...")
    train_data = {
        "filename": test_file,
        "model_type": "Isolation Forest"
    }
    
    try:
        response = requests.post(f"{base_url}/train", json=train_data)
        if response.status_code == 200:
            print("✓ Model trained successfully")
            print(f"   Features: {response.json().get('features', [])}")
        else:
            print(f"✗ Model training failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"✗ Model training error: {e}")
        return False
    
    # 2. Test comprehensive SHAP analysis
    print("\n2. Testing comprehensive SHAP analysis...")
    shap_data = {
        "filename": test_file,
        "model_type": "Isolation Forest"
    }
    
    try:
        response = requests.post(f"{base_url}/api/shap/comprehensive", json=shap_data)
        if response.status_code == 200:
            result = response.json()
            print("✓ Comprehensive SHAP analysis successful")
            print(f"   Summary plot: {'✓' if result.get('summary_plot') else '✗'}")
            print(f"   Importance plot: {'✓' if result.get('importance_plot') else '✗'}")
            print(f"   Dependence plots: {len(result.get('dependence_plots', {}))}")
            print(f"   Distribution plot: {'✓' if result.get('distribution_plot') else '✗'}")
            print(f"   Stats: {result.get('stats_summary', {})}")
        else:
            print(f"✗ Comprehensive SHAP analysis failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"✗ Comprehensive SHAP analysis error: {e}")
        return False
    
    # 3. Test anomaly analysis
    print("\n3. Testing anomaly analysis...")
    try:
        response = requests.post(f"{base_url}/api/shap/anomaly_analysis", json=shap_data)
        if response.status_code == 200:
            result = response.json()
            print("✓ Anomaly analysis successful")
            print(f"   Total anomalies: {result.get('total_anomalies', 0)}")
            print(f"   Anomaly percentage: {result.get('anomaly_percentage', 0):.2f}%")
            print(f"   Top anomalies: {len(result.get('top_anomalies', []))}")
            print(f"   Anomaly patterns: {len(result.get('anomaly_patterns', {}))}")
        else:
            print(f"✗ Anomaly analysis failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"✗ Anomaly analysis error: {e}")
        return False
    
    # 4. Test force plots
    print("\n4. Testing force plots...")
    force_data = {
        "filename": test_file,
        "model_type": "Isolation Forest",
        "indices": [0, 1, 2]  # Test with first 3 samples
    }
    
    try:
        response = requests.post(f"{base_url}/api/shap/force_plots", json=force_data)
        if response.status_code == 200:
            result = response.json()
            print("✓ Force plots successful")
            print(f"   Force plots generated: {len(result.get('force_plots', {}))}")
        else:
            print(f"✗ Force plots failed: {response.status_code}")
            print(response.text)
            return False
    except Exception as e:
        print(f"✗ Force plots error: {e}")
        return False
    
    print("\n✓ All SHAP tests passed!")
    return True

def test_data_loading():
    """Test if the data can be loaded and processed"""
    print("Testing data loading...")
    
    try:
        # Load the CSV file
        df = pd.read_csv("Uploads/kathmandu_cooperative_logs_20250702_101219.csv")
        print(f"✓ Data loaded successfully")
        print(f"   Rows: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Anomalies (label=1): {len(df[df['label'] == 1])}")
        print(f"   Normal transactions (label=0): {len(df[df['label'] == 0])}")
        
        # Check numeric features
        numeric_features = df.select_dtypes(include='number').columns.tolist()
        if 'label' in numeric_features:
            numeric_features.remove('label')
        print(f"   Numeric features: {numeric_features}")
        
        return True
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False

if __name__ == "__main__":
    print("SHAP Functionality Test")
    print("=" * 50)
    
    # Test data loading first
    if not test_data_loading():
        print("Data loading failed, cannot proceed with SHAP tests")
        exit(1)
    
    print("\n" + "=" * 50)
    
    # Test SHAP functionality
    if test_shap_functionality():
        print("\n🎉 All tests passed! SHAP functionality is working correctly.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.") 