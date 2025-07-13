# Comprehensive SHAP Analysis for Anomaly Detection

This document describes the comprehensive SHAP (SHapley Additive exPlanations) analysis features implemented for the Kathmandu Valley Cooperative anomaly detection system.

## Overview

The system now includes advanced SHAP visualizations and analysis tools that provide detailed insights into how the AI models make anomaly detection decisions. These features help users understand:

- Which features contribute most to anomaly detection
- How individual features influence the model's decisions
- Patterns in detected anomalies
- Detailed explanations for specific transactions

## Features

### 1. Comprehensive SHAP Analysis

**Endpoint:** `/api/shap/comprehensive`

Generates a complete SHAP analysis including:

- **Summary Plot (Beeswarm)**: Shows the distribution of SHAP values for each feature
- **Feature Importance Plot**: Bar chart of mean absolute SHAP values
- **Dependence Plots**: Shows how individual features affect SHAP values
- **Distribution Analysis**: Compares SHAP value distributions between normal and anomalous transactions
- **Interactive Charts**: Dynamic visualizations using Plotly

### 2. Anomaly Analysis

**Endpoint:** `/api/shap/anomaly_analysis`

Provides detailed analysis of detected anomalies:

- **Anomaly Summary Plot**: SHAP summary focused on detected anomalies
- **Top Anomalies Table**: Detailed information about the most significant anomalies
- **Anomaly Patterns**: Statistical analysis of feature patterns in anomalies
- **Anomaly Clustering**: Groups similar anomalies using PCA and K-means

### 3. Force Plots

**Endpoint:** `/api/shap/force_plots`

Generates force plots for individual transactions showing:

- How each feature contributes to the final prediction
- Visual representation of feature importance
- Detailed breakdown of anomaly scores

## Usage

### Web Interface

1. **Upload Data**: Upload your CSV file with transaction data
2. **Train Model**: Train either Isolation Forest or MLP model
3. **Navigate to SHAP Analysis**: Click on "SHAP Analysis" in the navigation
4. **Select Analysis Type**:
   - **Comprehensive Analysis**: Full SHAP analysis with all visualizations
   - **Anomaly Analysis**: Focused analysis of detected anomalies
   - **Force Plots**: Individual transaction explanations
5. **Generate Analysis**: Click "Generate Analysis" to create visualizations
6. **Export Report**: Download comprehensive analysis report

### API Usage

#### Comprehensive Analysis
```python
import requests

response = requests.post('http://localhost:5000/api/shap/comprehensive', json={
    'filename': 'your_data.csv',
    'model_type': 'Isolation Forest'
})

data = response.json()
# data contains summary_plot, importance_plot, dependence_plots, etc.
```

#### Anomaly Analysis
```python
response = requests.post('http://localhost:5000/api/shap/anomaly_analysis', json={
    'filename': 'your_data.csv',
    'model_type': 'Isolation Forest'
})

data = response.json()
# data contains anomaly_summary_plot, top_anomalies, anomaly_patterns, etc.
```

#### Force Plots
```python
response = requests.post('http://localhost:5000/api/shap/force_plots', json={
    'filename': 'your_data.csv',
    'model_type': 'Isolation Forest',
    'indices': [0, 1, 2]  # Specific sample indices
})

data = response.json()
# data contains force_plots for each specified index
```

## Visualizations

### 1. Summary Plot (Beeswarm)
- **Purpose**: Shows the distribution of SHAP values for each feature
- **Interpretation**: 
  - Each point represents a transaction
  - Color indicates feature value (red=high, blue=low)
  - Position on x-axis shows SHAP impact
  - Features are sorted by importance

### 2. Feature Importance Plot
- **Purpose**: Shows which features most influence anomaly detection
- **Interpretation**: Longer bars indicate more important features

### 3. Dependence Plots
- **Purpose**: Shows how individual features affect SHAP values
- **Interpretation**: 
  - X-axis: Feature value
  - Y-axis: SHAP value
  - Color: Another feature's value
  - Shows non-linear relationships

### 4. Distribution Analysis
- **Purpose**: Compares SHAP value distributions between normal and anomalous transactions
- **Interpretation**: Helps identify which features best distinguish anomalies

### 5. Force Plots
- **Purpose**: Shows how each feature contributes to a specific prediction
- **Interpretation**: 
  - Red bars: Increase anomaly score
  - Blue bars: Decrease anomaly score
  - Bar length: Magnitude of contribution

## Data Requirements

The system expects CSV files with the following columns:

- `log_id`: Unique transaction identifier
- `timestamp`: Transaction timestamp
- `member_id`: Member identifier
- `account_no`: Account number
- `transaction_type`: Type of transaction
- `amount`: Transaction amount
- `balance`: Account balance
- `branch_name`: Branch name
- `device_ip`: Device IP address
- `location`: Transaction location
- `label`: Anomaly label (0=normal, 1=anomaly) - optional

## Model Support

The SHAP analysis supports both:

1. **Isolation Forest**: Unsupervised anomaly detection
2. **MLP Model**: Supervised neural network (requires labels)

## Technical Details

### Dependencies
- `shap`: SHAP library for model explanations
- `matplotlib`: Plot generation
- `seaborn`: Enhanced visualizations
- `plotly`: Interactive charts
- `scikit-learn`: Machine learning models
- `pandas`: Data manipulation
- `numpy`: Numerical computations

### Performance Considerations
- SHAP analysis can be computationally intensive for large datasets
- The system uses efficient sampling for large datasets
- Results are cached to improve performance
- Background processing for long-running analyses

### Error Handling
- Graceful handling of missing data
- Fallback explanations when SHAP fails
- User-friendly error messages
- Detailed logging for debugging

## Testing

Run the test script to verify functionality:

```bash
python test_shap.py
```

This will test:
- Data loading
- Model training
- Comprehensive SHAP analysis
- Anomaly analysis
- Force plot generation

## Troubleshooting

### Common Issues

1. **"No anomalies detected"**
   - Check if your model is trained
   - Verify data contains anomalies
   - Try adjusting model parameters

2. **"SHAP analysis failed"**
   - Ensure all required dependencies are installed
   - Check data format and quality
   - Verify model file exists

3. **"File not found"**
   - Ensure CSV file is in the Uploads directory
   - Check file permissions
   - Verify file path

### Performance Optimization

1. **For large datasets**:
   - Use sampling for initial analysis
   - Focus on specific time periods
   - Use simpler models for faster results

2. **For real-time analysis**:
   - Pre-compute SHAP values
   - Use cached results
   - Implement background processing

## Future Enhancements

Planned improvements include:

1. **Real-time SHAP analysis**: Live monitoring of transactions
2. **Advanced clustering**: More sophisticated anomaly grouping
3. **Custom visualizations**: User-defined chart types
4. **Batch processing**: Handle multiple files simultaneously
5. **Export options**: PDF reports, Excel exports
6. **API rate limiting**: Better handling of concurrent requests

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review error logs
3. Test with sample data
4. Contact the development team

---

*This documentation covers the comprehensive SHAP analysis features implemented for the Kathmandu Valley Cooperative anomaly detection system.* 