import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

NUM_RECORDS = 2000

# Tier 1, 2, and 3 Cities Integration
cities = {
    # Tier 1
    "Delhi": {"lat_range": (28.40, 28.80), "lon_range": (76.80, 77.30), "weight": 0.15},
    "Mumbai": {"lat_range": (18.90, 19.20), "lon_range": (72.80, 73.00), "weight": 0.12},
    "Bangalore": {"lat_range": (12.80, 13.10), "lon_range": (77.50, 77.70), "weight": 0.10},
    "Chennai": {"lat_range": (12.90, 13.20), "lon_range": (80.10, 80.30), "weight": 0.08},
    "Hyderabad": {"lat_range": (17.30, 17.50), "lon_range": (78.30, 78.50), "weight": 0.08},
    "Pune": {"lat_range": (18.40, 18.60), "lon_range": (73.70, 73.90), "weight": 0.07},
    "Kolkata": {"lat_range": (22.50, 22.70), "lon_range": (88.30, 88.50), "weight": 0.07},
    
    # Tier 2
    "Jaipur": {"lat_range": (26.80, 27.00), "lon_range": (75.70, 75.90), "weight": 0.05},
    "Lucknow": {"lat_range": (26.70, 26.90), "lon_range": (80.80, 81.00), "weight": 0.05},
    "Nagpur": {"lat_range": (21.00, 21.20), "lon_range": (78.90, 79.20), "weight": 0.05},
    "Indore": {"lat_range": (22.60, 22.80), "lon_range": (75.70, 75.90), "weight": 0.04},
    
    # Tier 3
    "Nashik": {"lat_range": (19.90, 20.10), "lon_range": (73.70, 73.90), "weight": 0.04},
    "Madurai": {"lat_range": (9.80, 10.00), "lon_range": (78.00, 78.20), "weight": 0.03},
    "Varanasi": {"lat_range": (25.20, 25.40), "lon_range": (82.90, 83.10), "weight": 0.04},
    "Meerut": {"lat_range": (28.90, 29.10), "lon_range": (77.60, 77.80), "weight": 0.03}
}

city_names = list(cities.keys())
city_weights = [cities[c]["weight"] for c in city_names]

# Normalize weights to exactly 1.0 to prevent numpy errors
city_weights = [float(i)/sum(city_weights) for i in city_weights]

severities = ["Fatal", "Serious", "Minor"]
causes = ["Over-speeding", "Drunk Driving", "Distracted Driving", "Weather/Low Visibility", "Potholes", "Signal Jumping"]
road_types = ["Highway", "City Road", "Rural Road"]
weather_conds = ["Clear", "Rain", "Fog", "Dust Storm"]

start_date = datetime(2023, 1, 1)

data = []
for i in range(1, NUM_RECORDS + 1):
    city = np.random.choice(city_names, p=city_weights)
    lat = round(random.uniform(cities[city]["lat_range"][0], cities[city]["lat_range"][1]), 5)
    lon = round(random.uniform(cities[city]["lon_range"][0], cities[city]["lon_range"][1]), 5)
    
    hour = np.random.choice(
        [random.randint(0,5), random.randint(6,11), random.randint(12,17), random.randint(18,23)], 
        p=[0.1, 0.2, 0.3, 0.4]
    )
    acc_time = f"{hour:02d}:{random.randint(0, 59):02d}"
    acc_date = start_date + timedelta(days=random.randint(0, 364))
    
    cause = random.choice(causes)
    severity = np.random.choice(severities, p=[0.15, 0.35, 0.5])
    weather = random.choice(weather_conds)
    road_type = random.choice(road_types)
    
    # Logical Reasoning Enhancements
    
    # 1. Weather Logic Recheck: 
    # Previously, 100% of Rain/Fog led to "Weather/Low Visibility". This is flawed.
    # Bad weather increases the PROBABILITY of that cause, but Over-speeding or Drunk Driving still happens.
    if weather in ["Fog", "Rain", "Dust Storm"]:
        if random.random() < 0.6:  # 60% chance the cause was directly the weather
            cause = "Weather/Low Visibility"
            # If weather caused it, severity is often Serious rather than Fatal unless on a highway
            if road_type == "City Road":
                severity = np.random.choice(["Minor", "Serious"], p=[0.6, 0.4])
    else:
        # If it's Clear, Weather/Low Visibility shouldn't be the cause
        if cause == "Weather/Low Visibility":
            cause = random.choice(["Over-speeding", "Distracted Driving", "Signal Jumping"])
            
    # 2. Night Time Logic
    if hour >= 22 or hour <= 3:
        if random.random() < 0.35: 
            cause = "Drunk Driving"
            severity = "Fatal" if road_type == "Highway" else "Serious"
            
    # 3. Highway Speed Logic
    if road_type == "Highway" and cause == "Over-speeding":
        severity = "Fatal" if random.random() < 0.5 else "Serious"
        
    # 4. Pothole Logic (More likely on Rural/City roads than Highways, and worse in Rain)
    if cause == "Potholes":
        if road_type == "Highway":
            cause = "Distracted Driving" # Less potholes on Highways
        elif weather == "Rain" and random.random() < 0.7:
            severity = "Serious" # Hidden potholes in rain

    data.append([f"ID-{i}", city, lat, lon, acc_date.strftime("%Y-%m-%d"), acc_time, severity, cause, road_type, weather])

df = pd.DataFrame(data, columns=["Accident_ID", "Location", "Latitude", "Longitude", "Date", "Time", "Severity", "Cause", "Road_Type", "Weather"])
df.to_csv("data.csv", index=False)
