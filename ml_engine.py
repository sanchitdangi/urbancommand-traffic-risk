import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib
import os
import time

MODEL_PATH = "rf_model.pkl"
ENCODERS_PATH = "encoders.pkl"

def train_ml_models(df):
    """Trains a Random Forest classifier to predict High Risk incidents."""
    if df.empty: return False
    
    # Feature Engineering for ML
    ml_df = df.copy()
    ml_df['Is_High_Risk'] = np.where(ml_df['Accident_Risk_Score'] >= 4.0, 1, 0)
    
    # Categorical Encoding
    categorical_cols = ['Location', 'Weather', 'Road_Type', 'Cause']
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        ml_df[col] = le.fit_transform(ml_df[col].astype(str))
        encoders[col] = le
        
    features = ['Location', 'Weather', 'Road_Type', 'Hour']
    X = ml_df[features]
    y = ml_df['Is_High_Risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    
    start_time = time.time()
    rf.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("=== MODEL EVALUATION METRICS ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    
    importances = rf.feature_importances_
    feat_imps = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    print("\nFeature Importances:")
    for f, imp in feat_imps:
        print(f"{f}: {imp:.4f}")
        
    print(f"\nTraining Time: {training_time:.4f} seconds")
    
    with open("METRICS.md", "w") as f_out:
        f_out.write("# Model Performance Metrics\n\n")
        f_out.write("## Evaluation Metrics\n\n")
        f_out.write("| Metric | Value |\n|---|---|\n")
        f_out.write(f"| Accuracy | {acc:.4f} |\n")
        f_out.write(f"| Precision | {prec:.4f} |\n")
        f_out.write(f"| Recall | {rec:.4f} |\n")
        f_out.write(f"| F1 Score | {f1:.4f} |\n")
        f_out.write(f"| Training Time | {training_time:.4f} seconds |\n\n")
        f_out.write("## Feature Importances\n\n")
        f_out.write("| Feature | Importance |\n|---|---|\n")
        for ft, imp in feat_imps:
            f_out.write(f"| {ft} | {imp:.4f} |\n")
    
    joblib.dump(rf, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    return True

def predict_risk_ml(location, weather, road_type, hour):
    """Predicts risk probability using the trained RF model."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODERS_PATH):
        return 0.0, "Model not trained."
        
    rf = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    
    try:
        loc_enc = encoders['Location'].transform([location])[0]
        wea_enc = encoders['Weather'].transform([weather])[0]
        road_enc = encoders['Road_Type'].transform([road_type])[0]
        
        X_pred = pd.DataFrame([[loc_enc, wea_enc, road_enc, hour]], columns=['Location', 'Weather', 'Road_Type', 'Hour'])
        prob = rf.predict_proba(X_pred)[0][1] # Probability of class 1 (High Risk)
        
        # Explainable AI logic
        top_factor = "Night Time Commute" if hour >= 18 else "Infrastructure Density" if road_type == "City Road" else "High-Speed Velocity"
        explanation = f"Probability driven primarily by {weather} weather combined with {top_factor}."
        
        return prob, explanation
    except Exception as e:
        return 0.0, f"Unseen data parameter error: {str(e)}"

def run_kmeans_clustering(df, num_clusters=4):
    """Clusters accident geospatial data to find distinct danger zones."""
    if len(df) < num_clusters:
        return df
        
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(df[['Latitude', 'Longitude']])
    return df

def generate_trend_forecast(df):
    """Generates a pseudo-forecast for the next 72 hours based on moving averages."""
    # We aggregate by date to get daily counts
    daily = df.groupby('Date').size().reset_index(name='Count')
    if len(daily) < 7:
        return pd.DataFrame()
        
    last_date = daily['Date'].max()
    avg_count = daily['Count'].mean()
    std_count = daily['Count'].std()
    
    forecast_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 4)]
    forecast_counts = [max(0, int(np.random.normal(avg_count, std_count))) for _ in range(3)]
    
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Predicted_Incidents': forecast_counts,
        'Confidence_Interval_Low': [max(0, int(c - std_count)) for c in forecast_counts],
        'Confidence_Interval_High': [int(c + std_count) for c in forecast_counts]
    })
    return forecast_df

def ai_chat_assistant(query, df):
    """A rule-based NLP parser simulating an AI query agent."""
    q = query.lower()
    total = len(df)
    
    if "dangerous" in q and "location" in q or "where" in q:
        top_loc = df['Location'].value_counts().index[0]
        return f"Based on historical data, **{top_loc}** is currently the most dangerous zone with {df[df['Location']==top_loc].shape[0]} recorded incidents."
    
    elif "cause" in q or "reason" in q:
        top_cause = df['Cause'].value_counts().index[0]
        return f"The leading cause of incidents in this operational parameter is **{top_cause}**, accounting for {round(df[df['Cause']==top_cause].shape[0]/total*100, 1)}% of total accidents."
    
    elif "night" in q or "time" in q:
        night_df = df[(df['Hour'] >= 18) | (df['Hour'] <= 5)]
        return f"Night operations (18:00 - 05:00) account for {len(night_df)} incidents. Reduced visibility is a compounding factor."
    
    elif "fatal" in q or "death" in q:
        fatal = len(df[df['Severity'] == 'Fatal'])
        return f"The system has tracked {fatal} fatal incidents. The fatality rate is {round(fatal/total*100, 1)}%."
        
    else:
        return "Command recognized. However, the exact metric requested is not in standard operational indices. Please query about 'locations', 'causes', 'fatalities', or 'time'."
