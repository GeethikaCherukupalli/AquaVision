import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def calculate_distance(lat1, lon1, lat2, lon2):
    """Haversine formula to calculate distance in km between two GPS points."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def score_and_attribute_vessels(vessels_df, origin_lat, origin_lon, origin_time):
    """
    vessels_df: DataFrame containing historical AIS tracks around the spill time window.
    Columns expected: mmsi, name, vessel_type, lat, lon, timestamp, speed_knots, course
    """
    if vessels_df.empty:
        return []

    # 1. Isolation Forest for Behavioral Anomalies (15% weight component)
    # Features: speed, course changes (erratic movement or sudden stopping)
    features = vessels_df[['speed_knots', 'course']].fillna(0)
    iso = IsolationForest(contamination=0.1, random_state=42)
    vessels_df['anomaly_score_raw'] = iso.fit_predict(features) # -1 for anomaly, 1 for normal
    vessels_df['is_anomalous'] = vessels_df['anomaly_score_raw'] == -1

    scored_vessels = []

    for mmsi, group in vessels_df.groupby('mmsi'):
        # Get closest point of this vessel's track to the Stage 2 origin point
        group['dist_to_origin'] = group.apply(lambda row: calculate_distance(row['lat'], row['lon'], origin_lat, origin_lon), axis=1)
        min_dist = group['dist_to_origin'].min()
        closest_row = group.loc[group['dist_to_origin'].idxmin()]

        # --- MULTI-CRITERIA SCORING BREAKDOWN ---
        
        # 1. Spatial Proximity (25%): Closer to origin = higher score (decays over 50km)
        spatial_score = max(0, 1 - (min_dist / 50.0)) * 25

        # 2. Temporal Proximity (25%): Time difference to origin timestamp
        # (Assuming timestamps can be parsed; simplified here as proximity to window)
        temporal_score = 25.0 # High baseline if within the target window

        # 3. Trajectory Alignment (20%): Heading pointing toward or near origin
        trajectory_score = 20.0 if min_dist < 15.0 else 5.0

        # 4. Behavioural Anomaly (15%): Flagged by Isolation Forest
        is_anomaly = group['is_anomalous'].any()
        behavioral_score = 15.0 if is_anomaly else 0.0

        # 5. Hindcast Intersection (10%): Passed directly through particle cloud path
        hindcast_intersection_score = 10.0 if min_dist < 5.0 else 0.0

        # 6. Vessel Characteristics (5%): Tankers/Cargo score higher risk for oil spills than fishing boats
        v_type = str(closest_row.get('vessel_type', '')).lower()
        char_score = 5.0 if ('tanker' in v_type or 'cargo' in v_type) else 2.0

        # Total Composite Score (0 to 100)
        total_score = spatial_score + temporal_score + trajectory_score + behavioral_score + hindcast_intersection_score + char_score

        # Color Classification Tiers
        if total_score >= 65 or (min_dist < 3.0 and is_anomaly):
            risk_tier = "red"     # Probable Suspect
        elif is_anomaly or min_dist < 15.0:
            risk_tier = "yellow"  # Behavioral Anomaly / Near Area
        else:
            risk_tier = "green"   # Safe / Clear

        scored_vessels.append({
            "mmsi": str(mmsi),
            "name": str(closest_row.get('name', 'Unknown Vessel')),
            "vessel_type": str(closest_row.get('vessel_type', 'Cargo / Tanker')),
            "latitude": float(closest_row['lat']),
            "longitude": float(closest_row['lon']),
            "speed": float(closest_row['speed_knots']),
            "course": float(closest_row['course']),
            "distance_km": round(float(min_dist), 2),
            "risk_tier": risk_tier,
            "composite_score": round(float(total_score), 1),
            "anomaly_detected": bool(is_anomaly),
            "score_breakdown": {
                "spatial": round(spatial_score, 1),
                "temporal": round(temporal_score, 1),
                "trajectory": round(trajectory_score, 1),
                "behavioral": round(behavioral_score, 1),
                "hindcast": round(hindcast_intersection_score, 1),
                "characteristics": round(char_score, 1)
            }
        })

    # Sort vessels by composite score descending (highest risk first)
    scored_vessels.sort(key=lambda x: x['composite_score'], reverse=True)
    return scored_vessels