import pandas as pd
import numpy as np

def load_and_preprocess(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['Latitude', 'Longitude'])
    df['Weather'] = df['Weather'].fillna('Unknown')
    df['Cause'] = df['Cause'].fillna('Unknown')
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    time_objs = pd.to_datetime(df['Time'], format='%H:%M')
    df['Time'] = time_objs.dt.time
    df['Hour'] = time_objs.dt.hour
    
    df['Month'] = df['Date'].dt.month_name()
    df['Day/Night'] = np.where((df['Hour'] >= 6) & (df['Hour'] < 18), 'Day', 'Night')
    df['Weekday'] = df['Date'].dt.day_name()
    df['Weekend/Weekday'] = np.where(df['Date'].dt.dayofweek >= 5, 'Weekend', 'Weekday')
    
    severity_weights = {'Fatal': 3, 'Serious': 2, 'Minor': 1}
    df['Severity_Weight'] = df['Severity'].map(severity_weights)
    
    def get_time_weight(h):
        if 18 <= h <= 22: return 1.5
        elif h > 22 or h < 6: return 1.2
        return 1.0
    df['Time_Weight'] = df['Hour'].apply(get_time_weight)
    df['Accident_Risk_Score'] = df['Severity_Weight'] * df['Time_Weight']
    
    return df

def calculate_hotspot_risk(df):
    if df.empty:
        return pd.DataFrame()
        
    loc_stats = df.groupby('Location').agg(
        Frequency=('Accident_ID', 'count'),
        Avg_Severity=('Severity_Weight', 'mean'),
        Avg_Time_Weight=('Time_Weight', 'mean'),
        Total_Risk=('Accident_Risk_Score', 'sum'),
        Latitude=('Latitude', 'mean'),
        Longitude=('Longitude', 'mean')
    ).reset_index()
    
    def classify_risk(score):
        if score > 50: return 'High'
        elif score > 20: return 'Medium'
        return 'Low'
        
    loc_stats['Risk_Level'] = loc_stats['Total_Risk'].apply(classify_risk)
    return loc_stats

def simulate_risk(location_data, hour, weather, road_type):
    risk = 1.0
    if 18 <= hour <= 22: risk *= 1.5
    elif hour > 22 or hour < 6: risk *= 1.2
    if weather in ['Fog', 'Rain']: risk *= 1.4
    if road_type == 'Highway': risk *= 1.3
    
    if risk >= 2.0: return "High"
    elif risk >= 1.4: return "Medium"
    else: return "Low"

def intervention_engine(df_filtered):
    if df_filtered.empty:
        return None
        
    freq = len(df_filtered)
    fatal_ratio = len(df_filtered[df_filtered['Severity'] == 'Fatal']) / freq if freq > 0 else 0
    top_cause = df_filtered['Cause'].mode()[0] if not df_filtered['Cause'].empty else "Unknown"
    top_weather = df_filtered['Weather'].mode()[0]
    top_road = df_filtered['Road_Type'].mode()[0]
    
    pattern = ""
    cause_desc = ""
    intervention = ""
    priority = "Low"
    
    if freq > 50 and fatal_ratio > 0.2:
        priority = "High"
        pattern = f"High density of fatal accidents on {top_road}s."
        if top_cause == "Over-speeding":
            cause_desc = "Overspeeding under minimal enforcement."
            intervention = "Install automated speed cameras and heavy fines."
        elif top_cause == "Drunk Driving":
            cause_desc = "Intoxicated driving, likely late night."
            intervention = "Deploy police breathalyzer checkpoints."
        elif top_weather in ["Fog", "Rain"]:
            cause_desc = f"Low visibility due to {top_weather}."
            intervention = "Install reflective lane markers, cat eyes, and fog warning signs."
        else:
            cause_desc = f"Frequent {top_cause} incidents."
            intervention = "Conduct full traffic safety audit and improve signage."
    elif freq > 20:
        priority = "Medium"
        pattern = "Moderate accident frequency."
        cause_desc = f"Driven primarily by {top_cause}."
        if top_cause == "Signal Jumping":
            intervention = "Optimize signal timing and add red-light cameras."
        elif top_cause == "Potholes":
            intervention = "Immediate road maintenance and patching."
        else:
            intervention = "Increase visible traffic police presence."
    else:
        priority = "Low"
        pattern = "Low to normal accident rates."
        cause_desc = f"Occasional incidents of {top_cause}."
        intervention = "Maintain current monitoring."
        
    return {
        "pattern": pattern,
        "cause": cause_desc,
        "intervention": intervention,
        "priority": priority
    }
